from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness check endpoint returning HTTP 200 with status ok."""
    return {"status": "ok"}
