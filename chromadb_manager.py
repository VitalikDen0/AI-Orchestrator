"""ChromaDB vector store manager for conversation memory and user preferences."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, cast

from config import (
    CHROMA_DB_PATH,
    CHROMADB_EMBEDDING_MODEL,
    CHROMADB_USE_GPU_BY_DEFAULT,
    CHROMADB_DEFAULT_COLLECTION_NAME,
    CHROMADB_DEFAULT_COLLECTION_METADATA,
    CHROMADB_BACKGROUND_COLLECTION_NAME,
    DEFAULT_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def load_chromadb(embedding_model: str = CHROMADB_EMBEDDING_MODEL):
    """Load ChromaDB with specified embedding model."""
    try:
        logger.info("Loading ChromaDB components...")
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name=CHROMADB_BACKGROUND_COLLECTION_NAME,
            metadata=CHROMADB_DEFAULT_COLLECTION_METADATA,
        )

        model = SentenceTransformer(embedding_model)
        logger.info("ChromaDB loaded successfully")
        return {"client": client, "collection": collection, "model": model}
    except Exception as e:
        logger.error("Failed to load ChromaDB: %s", e, exc_info=True)
        return None


def get_background_loader():
    """Get singleton background loader instance (must be imported from elsewhere)."""
    # This is a placeholder - actual implementation should be in a separate module
    # For now, return a mock object
    class MockLoader:
        def __init__(self):
            self.loading_tasks = {}

        def start_loading(self, key, func, *args):
            pass

        def get_component(self, key, timeout=30):
            return None

    return MockLoader()


class ChromaDBManager:
    """Vector store manager for conversation memory and user preferences."""

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        embedding_model: str = CHROMADB_EMBEDDING_MODEL,
        use_gpu: bool = CHROMADB_USE_GPU_BY_DEFAULT,
    ) -> None:
        """Initialize ChromaDB manager.

        Args:
            db_path: Path to ChromaDB database.
            embedding_model: Model for creating embeddings (784 dimensions).
            use_gpu: Use GPU for embeddings if available.
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.use_gpu = use_gpu
        self.client: Any = None
        self.collection: Any = None
        self.embedding_model_obj: Any = None
        self.initialized = False
        self._sync_attempted = False

        os.makedirs(db_path, exist_ok=True)
        logger.debug(
            "ChromaDBManager initialized: db_path=%s, model=%s, use_gpu=%s",
            db_path,
            embedding_model,
            use_gpu,
        )

        # Start background initialization
        self._start_background_initialization()

    def _start_background_initialization(self) -> None:
        """Start background initialization of ChromaDB."""
        loader = get_background_loader()
        loader.start_loading("chromadb", load_chromadb, self.embedding_model)
        logger.debug("Background ChromaDB initialization started")

    def _ensure_initialized(self, timeout: int = 30) -> bool:
        """Ensure components are initialized."""
        if self.initialized:
            return True

        loader = get_background_loader()
        future = (
            loader.loading_tasks.get("chromadb")
            if hasattr(loader, "loading_tasks")
            else None
        )

        # Check if background loading completed
        if future and future.done():
            try:
                chromadb_data = future.result()
            except Exception as exc:
                logger.error("Background ChromaDB initialization failed: %s", exc)
                if not self._sync_attempted:
                    self._sync_attempted = True
                    return self._initialize_chromadb_sync()
                return False

            if chromadb_data:
                self.client = chromadb_data["client"]
                self.collection = chromadb_data["collection"]
                self.embedding_model_obj = chromadb_data["model"]
                self.initialized = True
                logger.info("ChromaDB initialized from background task")
                return True

            # Background loading completed without result
            if not self._sync_attempted:
                self._sync_attempted = True
                return self._initialize_chromadb_sync()
            return False

        # Background loading still in progress
        if future and not future.done():
            logger.info("ChromaDB still initializing in background, skipping memory usage")
            return False

        # No background loading - try sync initialization once
        if not self._sync_attempted:
            self._sync_attempted = True
            return self._initialize_chromadb_sync()
        return False

    def _initialize_chromadb_sync(self) -> bool:
        """Synchronous ChromaDB initialization as fallback."""
        try:
            logger.info("Synchronous ChromaDB initialization...")
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            self.collection = self.client.get_or_create_collection(
                name=CHROMADB_DEFAULT_COLLECTION_NAME,
                metadata=CHROMADB_DEFAULT_COLLECTION_METADATA,
            )

            self.embedding_model_obj = SentenceTransformer(self.embedding_model)
            self.initialized = True
            logger.info("ChromaDB initialized synchronously")
            return True

        except Exception as e:
            logger.error("Synchronous ChromaDB initialization failed: %s", e, exc_info=True)
            return False

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information for ChromaDB."""
        gpu_info: Dict[str, Any] = {
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory": None,
            "device_used": "cpu",
        }

        try:
            if TORCH_AVAILABLE and torch and torch.cuda.is_available():
                gpu_info["gpu_available"] = True
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                gpu_info["gpu_memory"] = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )
                gpu_info["device_used"] = "cuda" if self.use_gpu else "cpu"

                logger.info("GPU available: %s", gpu_info["gpu_name"])
                logger.info("GPU memory: %.1f GB", gpu_info["gpu_memory"])
            else:
                logger.info("GPU unavailable, using CPU")

        except Exception as e:
            logger.warning("Error getting GPU info: %s", e)

        return gpu_info

    def _encode_text_to_embedding(self, text: str) -> List[float]:
        """Convert text to embedding vector."""
        if self.embedding_model_obj is None:
            return []

        try:
            vector = self.embedding_model_obj.encode(text, convert_to_numpy=True)
            vector_any: Any = vector

            if hasattr(vector_any, "tolist"):
                return list(vector_any.tolist())

            return [float(value) for value in cast(Iterable[float], vector_any)]
        except Exception as e:
            logger.error("Failed to create embedding: %s", e)
            return []

    def add_conversation_memory(
        self,
        user_message: str,
        ai_response: str,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force_add: bool = False,
    ) -> bool:
        """Add conversation to vector store.

        Args:
            user_message: User's message.
            ai_response: AI's response.
            context: Additional context.
            metadata: Additional metadata.
            force_add: Force add without duplicate check.

        Returns:
            True if successfully added, False on error or duplicate.
        """
        if not self._ensure_initialized():
            return False

        try:
            # Check for duplicates unless force_add
            if not force_add:
                logger.debug("Checking for duplicates: '%s...'", user_message[:50])
                similar = self.search_similar_conversations(
                    user_message, n_results=1, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
                )

                if similar and len(similar) > 0:
                    similarity = similar[0].get("similarity", 0)
                    logger.debug("Found similar conversation with similarity=%.3f", similarity)

                    if similarity > DEFAULT_SIMILARITY_THRESHOLD:
                        logger.info("Duplicate found with similarity=%.3f, skipping", similarity)
                        return False
                else:
                    logger.debug("No similar conversations found, safe to add")
            else:
                logger.debug("Force add enabled, skipping duplicate check")

            # Create unique ID
            import uuid

            timestamp = int(time.time())
            unique_suffix = str(uuid.uuid4())[:8]
            record_id = f"conv_{timestamp}_{unique_suffix}"

            # Combine text for embedding
            combined_text = f"User: {user_message}\nAI: {ai_response}"
            if context:
                combined_text += f"\nContext: {context}"

            # Create embedding
            if not self.initialized or self.embedding_model_obj is None:
                logger.warning("Embedding model not initialized, skipping ChromaDB add")
                return False

            embedding = self._encode_text_to_embedding(combined_text)
            if not embedding:
                logger.warning("Failed to create embedding for conversation")
                return False

            # Prepare metadata
            record_metadata: Dict[str, Any] = {
                "timestamp": timestamp,
                "user_message": user_message,
                "ai_response": ai_response,
                "context": context,
                "type": "conversation",
            }

            if metadata:
                record_metadata.update(metadata)

            # Add to collection
            if self.collection is None:
                logger.warning("ChromaDB collection not initialized")
                return False

            # Check if ID already exists (unlikely with UUID)
            try:
                existing = self.collection.get(ids=[record_id])
                if existing and existing.get("ids") and len(existing["ids"]) > 0:
                    record_id = f"conv_{timestamp}_{unique_suffix}_{hash(ai_response) % 1000}"
                    logger.info("ID already exists, using new ID: %s", record_id)
            except Exception:
                pass  # ID doesn't exist

            self.collection.add(
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[record_metadata],
                ids=[record_id],
            )

            # Log total count
            try:
                total_count = self.collection.count()
                logger.info("Added record to ChromaDB: %s (total: %d)", record_id, total_count)
            except Exception:
                logger.info("Added record to ChromaDB: %s", record_id)

            return True

        except Exception as e:
            logger.error("Error adding to ChromaDB: %s", e, exc_info=True)
            return False

    def search_similar_conversations(
        self,
        query: str,
        n_results: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar conversations in vector store.

        Args:
            query: Search query.
            n_results: Number of results.
            similarity_threshold: Similarity threshold (automatic if None).

        Returns:
            List of found conversations with metadata.
        """
        if not self._ensure_initialized():
            return []

        try:
            if not self.initialized or self.embedding_model_obj is None:
                logger.warning("Embedding model not initialized, search not possible")
                return []

            logger.info("Searching for similar conversations: '%s'", query)
            query_embedding = self._encode_text_to_embedding(query)
            if not query_embedding:
                logger.warning("Failed to create embedding for query")
                return []

            if self.collection is None:
                logger.warning("ChromaDB collection not available, search not possible")
                return []

            # Check total count before search
            try:
                total_count = self.collection.count()
                logger.info("Total records in ChromaDB: %d", total_count)
            except Exception as e:
                logger.warning("Failed to get record count: %s", e)

            # Increase results for better search
            search_results = max(n_results * 3, 15)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_results,
                where={"type": "conversation"},  # type: ignore[arg-type]
            )

            # Analyze results for adaptive threshold
            filtered_results: List[Dict[str, Any]] = []
            found_count = 0

            if isinstance(results, dict) and results:
                distances = results.get("distances")
                ids = results.get("ids")
                documents = results.get("documents")
                metadatas = results.get("metadatas")

                if (
                    distances
                    and isinstance(distances, list)
                    and distances
                    and distances[0]
                ):
                    logger.info("Processing %d search results", len(distances[0]))

                    # Calculate adaptive threshold if not specified
                    if similarity_threshold is None:
                        similarities = [1 - d for d in distances[0]]
                        if similarities:
                            max_sim = max(similarities)
                            avg_sim = sum(similarities) / len(similarities)

                            if max_sim > 0.1:
                                adaptive_threshold = min(
                                    avg_sim + 0.1, 0.3, max_sim - 0.05
                                )
                            else:
                                adaptive_threshold = -0.2

                            logger.info(
                                "Adaptive similarity threshold: %.3f (max: %.3f, avg: %.3f)",
                                adaptive_threshold,
                                max_sim,
                                avg_sim,
                            )
                        else:
                            adaptive_threshold = 0.1
                    else:
                        adaptive_threshold = similarity_threshold

                    for i, distance in enumerate(distances[0]):
                        similarity = 1 - distance

                        # Log first 3 results for debugging
                        if i < 3:
                            logger.info(
                                "   Result %d: similarity=%.3f, distance=%.3f",
                                i + 1,
                                similarity,
                                distance,
                            )

                        if similarity >= adaptive_threshold:
                            try:
                                idv = (
                                    ids[0][i]
                                    if ids and ids[0] and len(ids[0]) > i
                                    else None
                                )
                                doc = (
                                    documents[0][i]
                                    if documents
                                    and documents[0]
                                    and len(documents[0]) > i
                                    else None
                                )
                                meta = (
                                    metadatas[0][i]
                                    if metadatas
                                    and metadatas[0]
                                    and len(metadatas[0]) > i
                                    else None
                                )
                            except Exception as e:
                                logger.warning("Error extracting result %d: %s", i, e)
                                continue

                            result = {
                                "id": idv,
                                "document": doc,
                                "metadata": meta,
                                "similarity": similarity,
                                "distance": distance,
                            }
                            filtered_results.append(result)
                            found_count += 1

                            if found_count >= n_results:
                                break

                    # If nothing found with adaptive threshold, take best results
                    if not filtered_results and distances[0]:
                        logger.info(
                            "Nothing found with threshold %.3f, taking %d best results",
                            adaptive_threshold,
                            min(3, len(distances[0])),
                        )
                        best_results = min(3, len(distances[0]))
                        for i in range(best_results):
                            distance = distances[0][i]
                            similarity = 1 - distance

                            try:
                                idv = (
                                    ids[0][i]
                                    if ids and ids[0] and len(ids[0]) > i
                                    else None
                                )
                                doc = (
                                    documents[0][i]
                                    if documents
                                    and documents[0]
                                    and len(documents[0]) > i
                                    else None
                                )
                                meta = (
                                    metadatas[0][i]
                                    if metadatas
                                    and metadatas[0]
                                    and len(metadatas[0]) > i
                                    else None
                                )

                                result = {
                                    "id": idv,
                                    "document": doc,
                                    "metadata": meta,
                                    "similarity": similarity,
                                    "distance": distance,
                                }
                                filtered_results.append(result)
                            except Exception as e:
                                logger.warning("Error extracting best result %d: %s", i, e)
                else:
                    logger.warning("Empty search results in ChromaDB")
            else:
                logger.warning("Invalid search results format")

            logger.info("Found %d similar conversations", len(filtered_results))
            return filtered_results

        except Exception as e:
            logger.error("Error searching ChromaDB: %s", e, exc_info=True)
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.initialized:
            return {"error": "ChromaDB not initialized"}

        try:
            if self.collection is None:
                logger.warning("ChromaDB collection not available")
                return {"error": "ChromaDB not initialized"}

            total_count = self.collection.count()

            conversations = self.collection.get(where={"type": "conversation"})  # type: ignore[arg-type]
            preferences = self.collection.get(where={"type": "preference"})  # type: ignore[arg-type]

            conv_ids = conversations.get("ids") if isinstance(conversations, dict) else None
            pref_ids = preferences.get("ids") if isinstance(preferences, dict) else None

            stats = {
                "total_records": total_count,
                "conversations": len(conv_ids) if conv_ids else 0,
                "preferences": len(pref_ids) if pref_ids else 0,
                "database_path": self.db_path,
                "embedding_model": self.embedding_model,
            }

            logger.info("Database stats: %s", stats)
            return stats

        except Exception as e:
            logger.error("Error getting ChromaDB stats: %s", e, exc_info=True)
            return {"error": str(e)}

    def add_user_preference(
        self,
        preference_text: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add user preference to vector store."""
        if not self._ensure_initialized():
            logger.error("ChromaDB not initialized")
            return False

        try:
            if self.collection is None:
                return False

            meta = metadata or {}
            meta.update({
                "type": "preference",
                "category": category,
                "timestamp": str(datetime.now()),
            })

            doc_id = f"pref_{datetime.now().timestamp()}"
            self.collection.add(
                documents=[preference_text],
                metadatas=[meta],
                ids=[doc_id],
            )

            logger.info("Added user preference: category=%s", category)
            return True

        except Exception as e:
            logger.error("Error adding user preference: %s", e, exc_info=True)
            return False

    def get_conversation_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context from previous conversations."""
        if not self._ensure_initialized():
            logger.error("ChromaDB not initialized")
            return ""

        try:
            if self.collection is None:
                return ""

            results = self.search_similar_conversations(query, n_results=5, similarity_threshold=0.7)
            
            if not results:
                return ""

            context_parts = []
            current_length = 0

            for item in results:
                text = item.get("text", "")
                if current_length + len(text) > max_context_length:
                    break
                context_parts.append(text)
                current_length += len(text)

            context = "\n\n".join(context_parts)
            logger.info("Retrieved context: %d characters from %d conversations", len(context), len(context_parts))
            return context

        except Exception as e:
            logger.error("Error getting conversation context: %s", e, exc_info=True)
            return ""

    def get_user_preferences_summary(self, query: Optional[str] = None) -> str:
        """Get user preferences summary."""
        if not self._ensure_initialized():
            logger.error("ChromaDB not initialized")
            return ""

        try:
            if self.collection is None:
                return ""

            where_filter: Dict[str, Any] = {"type": "preference"}
            results = self.collection.get(where=where_filter, limit=20)  # type: ignore[arg-type]

            if not results or not isinstance(results, dict):
                return ""

            docs = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if not docs:
                return ""

            # Group by category
            by_category: Dict[str, List[str]] = {}
            for doc, meta in zip(docs, metadatas):
                if not isinstance(meta, dict):
                    continue
                cat = meta.get("category", "general")
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(doc)

            summary_parts = []
            for cat, prefs in by_category.items():
                summary_parts.append(f"{cat.upper()}: {', '.join(prefs[:5])}")

            summary = "\n".join(summary_parts)
            logger.info("Retrieved preferences summary: %d categories", len(by_category))
            return summary

        except Exception as e:
            logger.error("Error getting preferences summary: %s", e, exc_info=True)
            return ""

    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Remove old records from memory."""
        if not self._ensure_initialized():
            logger.error("ChromaDB not initialized")
            return 0

        try:
            if self.collection is None:
                return 0

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = cutoff_date.timestamp()

            # Get all records
            all_records = self.collection.get()
            if not isinstance(all_records, dict):
                return 0

            ids_to_delete = []
            metadatas = all_records.get("metadatas", [])
            record_ids = all_records.get("ids", [])

            for record_id, meta in zip(record_ids, metadatas):
                if not isinstance(meta, dict):
                    continue
                
                timestamp_str = meta.get("timestamp", "")
                try:
                    # Try parsing ISO format
                    record_date = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    if record_date.timestamp() < cutoff_timestamp:
                        ids_to_delete.append(record_id)
                except (ValueError, AttributeError):
                    # Skip records with invalid timestamps
                    continue

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d old records (older than %d days)", len(ids_to_delete), days_to_keep)

            return len(ids_to_delete)

        except Exception as e:
            logger.error("Error cleaning up old records: %s", e, exc_info=True)
            return 0


__all__ = ["ChromaDBManager", "load_chromadb"]
