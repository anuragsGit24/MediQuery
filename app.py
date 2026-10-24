import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from support_rag import RAGService, RAGBackendError

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("healthcare_rag")

rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    try:
        rag_service = RAGService()
        rag_service.initialize()
        logger.info("RAG service initialized")
    except Exception:
        logger.exception("RAG initialization failed")
        raise
    yield
    rag_service = None


app = FastAPI(
    title="Healthcare Knowledge RAG API",
    description=(
        "Experiment 7: a healthcare education RAG API built from the "
        "Experiment 5 TXT, PDF and DOCX knowledge base. This is educational "
        "information only, not medical advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="A question answerable from the supplied educational documents.",
        examples=["What is cashless hospitalization?"],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value


class SourceItem(BaseModel):
    file_name: str
    page: int | None = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    best_distance: float | None
    latency_ms: float
    grounded: bool


@app.get("/health", tags=["Operations"])
def health():
    return {"status": "ok", "service": "healthcare-rag-api"}


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(payload: AskRequest, request: Request):
    global rag_service
    started = time.perf_counter()

    if rag_service is None:
        logger.error("Request received before RAG service initialization")
        raise HTTPException(status_code=503, detail="RAG service is not ready.")

    try:
        answer, documents, best_distance, grounded = rag_service.answer_question(
            payload.question
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        sources = []
        seen = set()
        for doc in documents:
            metadata = doc.metadata
            file_name = metadata.get("file_name") or os.path.basename(
                metadata.get("source", "unknown")
            )
            page = metadata.get("page")
            page_number = int(page) + 1 if page is not None else None
            key = (file_name, page_number, doc.page_content[:100])
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceItem(
                    file_name=file_name,
                    page=page_number,
                    excerpt=doc.page_content[:400],
                )
            )

        logger.info(
            "ask_success path=%s grounded=%s latency_ms=%.2f source_count=%d",
            request.url.path, grounded, latency_ms, len(sources)
        )
        return AskResponse(
            answer=answer,
            sources=sources,
            best_distance=round(best_distance, 4) if best_distance is not None else None,
            latency_ms=latency_ms,
            grounded=grounded,
        )
    except RAGBackendError:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.warning("ask_backend_failure latency_ms=%.2f", latency_ms)
        raise HTTPException(
            status_code=502,
            detail="The answer service is temporarily unavailable. Please try again later.",
        )
    except HTTPException:
        raise
    except Exception:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception("ask_internal_failure latency_ms=%.2f", latency_ms)
        raise HTTPException(
            status_code=500,
            detail="The request could not be completed due to an internal error.",
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )
