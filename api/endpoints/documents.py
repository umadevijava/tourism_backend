from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.deps import SessionDep, VectorDatabaseDep
from backend.core.config import settings
from backend.schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse
from chatbot.bot.memory.document_registry import DocumentRegistry
from chatbot.bot.memory.vector_database.id_generator import generate_id
from chatbot.document_loader.loader import DirectoryLoader
from chatbot.helpers.log import get_logger
from chatbot.memory_builder import split_chunks

logger = get_logger(__name__)

router = APIRouter()

# In-memory store of the uploaded document metadata, keyed by document_id.
_uploaded_documents: dict[str, DocumentInfo] = {}


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
    responses={
        400: {"description": "Bad Request - Invalid file type."},
        409: {"description": "Conflict - Document with the same filename already exists."},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    index: VectorDatabaseDep,
    session: SessionDep,
):
    """
    Upload a document to the knowledge base.

    Args:
        file: The file to upload. Must have an allowed extension.
        index: Vector database dependency for storing document chunks.
        session: Database session dependency for the document registry.

    Returns:
        DocumentUploadResponse containing the generated document_id and filename.

    Raises:
        HTTPException: 400 if file type is not supported.
        HTTPException: 409 if a document with the same filename already exists.
    """
    logger.info(f"Starting upload for file: {file.filename} (size: {file.size} bytes)")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        logger.warning(f"Unsupported file type '{suffix}' attempted for upload")
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Allowed: {sorted(settings.ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    registry = DocumentRegistry(session)
    existing = registry.get_by_filename(file.filename or "")
    if existing is not None:
        logger.warning(f"Attempting to upload duplicate document: {file.filename}")
        raise HTTPException(
            status_code=409,
            detail=f"Document '{file.filename}' already exists.",
        )

    dest_dir = settings.DOCS_PATH
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file.filename
    document_id = generate_id(str(file_path))

    logger.info(f"Generated document_id: {document_id}, saving to: {file_path}")

    content = await file.read()
    file_path.write_bytes(content)
    logger.info(f"File saved successfully. Content size: {len(content)} bytes")

    # Use DirectoryLoader to load the file content (same as build_memory_index)
    # This ensures consistent content processing and version hashing
    # TODO: refactor to avoid writing to disk and re-reading, but this is simpler for now and leverages existing loader
    #  logic
    try:
        logger.info(f"Loading document content from: {file_path.name}")
        loader = DirectoryLoader(
            path=file_path.parent,
            glob=file_path.name,
            show_progress=False,
        )
        loaded_docs = loader.load()

        if not loaded_docs:
            logger.error(f"Failed to extract content from file: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load document '{file.filename}'. The file may be corrupted or in an"
                f" unsupported format.",
            )

        # Extract the loaded document (should be exactly one)
        document = loaded_docs[0]
        page_content = document.page_content
        logger.info(f"Successfully loaded document content. Content length: {len(page_content)} characters")

    except HTTPException:
        # Re-raise HTTPException as-is
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up file after error: {file_path}")
        raise
    except Exception as exc:
        logger.exception(
            f"Failed to load uploaded file '{file.filename}': {exc}",
        )
        # Clean up the saved file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process document '{file.filename}': {str(exc)}",
        )

    version_hash = generate_id(page_content)
    logger.info(f"Generated version_hash: {version_hash}")

    # Update document metadata with our tracking fields
    document.metadata.update(
        {
            "source": str(file_path),
            "document_id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "version_hash": version_hash,
        }
    )

    # Store the document metadata in the in-memory registry for listing purposes
    doc_info = DocumentInfo(
        document_id=document_id,
        filename=file.filename or document_id,
        size=len(content),
        content_type=file.content_type or "application/octet-stream",
        version_hash=version_hash,
    )
    _uploaded_documents[document_id] = doc_info

    # Split the document into chunks for vector indexing
    logger.info(f"Splitting document into chunks (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
    chunks = split_chunks([document], chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    # Inject document_id + version_hash into every chunk's metadata
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["version_hash"] = version_hash

    num_chunks = len(chunks)
    logger.info(f"Generated {num_chunks} chunks from document")
    logger.info("Adding document chunks to the vector database index...")

    try:
        chunk_ids = index.from_chunks(chunks)
        logger.info(f"Successfully added {len(chunk_ids)} chunks to vector index")
    except Exception as exc:
        logger.exception(f"Failed to add chunks to vector database: {exc}")
        # Clean up the document file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index document '{file.filename}': {str(exc)}",
        )

    try:
        registry.upsert(
            document_id,
            source=str(file_path),
            filename=file.filename or document_id,
            size=len(content),
            content_type=file.content_type or "application/octet-stream",
            version_hash=version_hash,
            chunk_ids=chunk_ids,
        )
        logger.info(f"Successfully registered document in database: {document_id}")
    except Exception as exc:
        logger.exception(f"Failed to register document in database: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register document in database: {str(exc)}",
        )

    logger.info(f"✓ Document upload completed successfully: {file.filename} (ID: {document_id}, Chunks: {num_chunks})")

    return DocumentUploadResponse(document_id=document_id, filename=file.filename or document_id)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(session: SessionDep):
    """
    List all uploaded documents.

    Returns:
        DocumentListResponse containing a list of all document metadata.
    """
    try:
        registry = DocumentRegistry(session)
        # Get all documents from database
        all_docs = registry.get_all()
        
        # Convert database records to DocumentInfo objects
        documents = [
            DocumentInfo(
                document_id=doc.document_id,
                filename=doc.filename,
                size=doc.size,
                content_type=doc.content_type,
                version_hash=doc.version_hash,
            )
            for doc in all_docs
        ]
        
        logger.info(f"Listing documents. Database registry: {len(documents)} documents")
        return DocumentListResponse(documents=documents)
    except Exception as exc:
        logger.error(f"Failed to list documents: {exc}")
        # Fallback to in-memory registry if database query fails
        logger.info(f"Falling back to in-memory registry: {len(_uploaded_documents)} documents")
        return DocumentListResponse(documents=list(_uploaded_documents.values()))


@router.delete(
    "/documents/{document_id}",
    status_code=204,
    responses={404: {"description": "Not Found - Document with the given ID does not exist."}},
)
async def delete_document(document_id: str, index: VectorDatabaseDep, session: SessionDep):
    """
    Delete the uploaded document from the knowledge base.

    Removes the document's metadata, associated file from disk, and its
    chunks from the vector database index.

    Args:
        document_id: The unique identifier of the document to delete.
        index: Vector database dependency for removing document chunks.
        session: Database session dependency for the document registry.

    Raises:
        HTTPException: 404 if the document with the given ID is not found.
    """
    logger.info(f"Attempting to delete document: {document_id}")
    
    registry = DocumentRegistry(session)
    entry = registry.get(document_id)

    if entry is None:
        logger.warning(f"Delete request for non-existent document: {document_id}")
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    logger.info(f"Deleting {len(entry.chunk_ids or [])} chunks from vector index...")
    index.delete_chunks_by_document_id(document_id, chunk_ids=entry.chunk_ids or None)
    
    logger.info(f"Removing document from registry...")
    registry.remove(document_id)

    file_path = settings.DOCS_PATH / entry.filename
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Deleted file from disk: {file_path}")
    
    if document_id in _uploaded_documents:
        del _uploaded_documents[document_id]
    
    logger.info(f"✓ Document deleted successfully: {document_id}")
