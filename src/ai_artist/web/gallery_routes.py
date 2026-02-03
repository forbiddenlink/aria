"""Community Gallery API routes - public sharing, likes, comments."""

import secrets
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from ..db.models import (
    CollectionArtwork,
    GalleryCollection,
    GalleryComment,
    GalleryLike,
    GalleryShare,
    GeneratedImage,
)
from ..db.session import get_db
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
router = APIRouter(prefix="/api/gallery", tags=["gallery"])

# Type alias for cleaner dependency injection
DbSession = Annotated[Session, Depends(get_db)]


# === Pydantic Models ===


class GalleryImageResponse(BaseModel):
    """Public gallery image response."""

    id: int
    share_id: str
    filename: str
    prompt: str
    model_id: str
    tags: list[str]
    created_at: str
    like_count: int
    comment_count: int
    share_count: int
    view_count: int
    is_featured: bool
    aesthetic_score: float | None = None
    metadata: dict | None = None


class GalleryListResponse(BaseModel):
    """Paginated list of gallery images."""

    total: int
    page: int
    per_page: int
    images: list[GalleryImageResponse]


class LikeResponse(BaseModel):
    """Response after liking/unliking an image."""

    success: bool
    liked: bool
    like_count: int


class CommentCreate(BaseModel):
    """Request to create a comment."""

    text: str = Field(..., min_length=1, max_length=500)
    display_name: str = Field(default="Anonymous", max_length=50)


class CommentResponse(BaseModel):
    """Single comment response."""

    id: int
    display_name: str
    text: str
    created_at: str


class CommentListResponse(BaseModel):
    """List of comments for an image."""

    total: int
    comments: list[CommentResponse]


class ShareRequest(BaseModel):
    """Request to track a share."""

    platform: str = Field(..., pattern="^(twitter|facebook|pinterest|link)$")


class ShareResponse(BaseModel):
    """Response after tracking a share."""

    success: bool
    share_count: int
    share_url: str


class PublishRequest(BaseModel):
    """Request to publish an image to gallery."""

    image_id: int


class PublishResponse(BaseModel):
    """Response after publishing."""

    success: bool
    share_id: str
    share_url: str


# === Helper Functions ===


def get_session_id(request: Request) -> str:
    """Get or create anonymous session ID from cookies/headers."""
    session_id = request.cookies.get("gallery_session")
    if not session_id:
        session_id = request.headers.get("X-Gallery-Session")
    if not session_id:
        session_id = secrets.token_urlsafe(32)
    return session_id


def generate_share_id() -> str:
    """Generate a short unique share ID."""
    return secrets.token_urlsafe(8)[:12]


def image_to_response(image: GeneratedImage) -> GalleryImageResponse:
    """Convert database model to response."""
    return GalleryImageResponse(
        id=image.id,
        share_id=image.share_id or "",
        filename=image.filename,
        prompt=image.prompt,
        model_id=image.model_id,
        tags=image.tags or [],
        created_at=image.created_at.isoformat() if image.created_at else "",
        like_count=image.like_count or 0,
        comment_count=image.comment_count or 0,
        share_count=image.share_count or 0,
        view_count=image.view_count or 0,
        is_featured=image.is_featured or False,
        aesthetic_score=image.aesthetic_score,
        metadata=image.generation_params,
    )


# === Routes ===


@router.get("/public", response_model=GalleryListResponse)
@limiter.limit("60/minute")
async def list_public_images(
    request: Request,
    db: DbSession,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort: str = Query("newest", pattern="^(newest|popular|trending|featured)$"),
    tag: str | None = None,
    mood: str | None = None,
    search: str | None = None,
):
    """List public gallery images with filtering and sorting."""
    query = db.query(GeneratedImage).filter(GeneratedImage.is_public.is_(True))  # noqa: E712

    # Apply filters
    if tag:
        query = query.filter(GeneratedImage.tags.contains([tag]))
    if mood:
        query = query.filter(GeneratedImage.generation_params["mood"].astext == mood)
    if search:
        query = query.filter(
            or_(
                GeneratedImage.prompt.ilike(f"%{search}%"),
                GeneratedImage.tags.contains([search]),
            )
        )

    # Apply sorting
    if sort == "newest":
        query = query.order_by(GeneratedImage.created_at.desc())
    elif sort == "popular":
        query = query.order_by(GeneratedImage.like_count.desc())
    elif sort == "trending":
        # Trending: recent + engagement
        query = query.order_by(
            (GeneratedImage.like_count + GeneratedImage.view_count).desc(),
            GeneratedImage.created_at.desc(),
        )
    elif sort == "featured":
        query = query.filter(GeneratedImage.is_featured == True).order_by(  # noqa: E712
            GeneratedImage.created_at.desc()
        )

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * per_page
    images = query.offset(offset).limit(per_page).all()

    return GalleryListResponse(
        total=total,
        page=page,
        per_page=per_page,
        images=[image_to_response(img) for img in images],
    )


@router.get("/image/{share_id}", response_model=GalleryImageResponse)
@limiter.limit("120/minute")
async def get_image_by_share_id(
    request: Request,
    share_id: str,
    db: DbSession,
):
    """Get a single image by its share ID and increment view count."""
    image = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.share_id == share_id,
            GeneratedImage.is_public.is_(True),  # noqa: E712
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Increment view count
    image.view_count = (image.view_count or 0) + 1
    db.commit()

    return image_to_response(image)


@router.post("/image/{share_id}/like", response_model=LikeResponse)
@limiter.limit("30/minute")
async def toggle_like(
    request: Request,
    share_id: str,
    db: DbSession,
):
    """Like or unlike an image (toggle)."""
    session_id = get_session_id(request)

    image = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.share_id == share_id,
            GeneratedImage.is_public.is_(True),  # noqa: E712
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check if already liked
    existing_like = (
        db.query(GalleryLike)
        .filter(
            GalleryLike.image_id == image.id,
            GalleryLike.session_id == session_id,
        )
        .first()
    )

    if existing_like:
        # Unlike
        db.delete(existing_like)
        image.like_count = max(0, (image.like_count or 0) - 1)
        liked = False
    else:
        # Like
        new_like = GalleryLike(image_id=image.id, session_id=session_id)
        db.add(new_like)
        image.like_count = (image.like_count or 0) + 1
        liked = True

    db.commit()

    return LikeResponse(
        success=True,
        liked=liked,
        like_count=image.like_count or 0,
    )


@router.get("/image/{share_id}/like-status")
@limiter.limit("120/minute")
async def get_like_status(
    request: Request,
    share_id: str,
    db: DbSession,
):
    """Check if current session has liked an image."""
    session_id = get_session_id(request)

    image = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.share_id == share_id,
            GeneratedImage.is_public.is_(True),  # noqa: E712
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    existing_like = (
        db.query(GalleryLike)
        .filter(
            GalleryLike.image_id == image.id,
            GalleryLike.session_id == session_id,
        )
        .first()
    )

    return {"liked": existing_like is not None}


@router.post("/image/{share_id}/comment", response_model=CommentResponse)
@limiter.limit("10/minute")
async def add_comment(
    request: Request,
    share_id: str,
    comment: CommentCreate,
    db: DbSession,
):
    """Add a comment to an image."""
    session_id = get_session_id(request)

    image = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.share_id == share_id,
            GeneratedImage.is_public.is_(True),  # noqa: E712
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    new_comment = GalleryComment(
        image_id=image.id,
        session_id=session_id,
        display_name=comment.display_name.strip() or "Anonymous",
        text=comment.text.strip(),
    )
    db.add(new_comment)
    image.comment_count = (image.comment_count or 0) + 1
    db.commit()
    db.refresh(new_comment)

    return CommentResponse(
        id=new_comment.id,
        display_name=new_comment.display_name,
        text=new_comment.text,
        created_at=new_comment.created_at.isoformat(),
    )


@router.get("/image/{share_id}/comments", response_model=CommentListResponse)
@limiter.limit("60/minute")
async def get_comments(
    request: Request,
    share_id: str,
    db: DbSession,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=50),
):
    """Get comments for an image."""
    image = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.share_id == share_id,
            GeneratedImage.is_public.is_(True),  # noqa: E712
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    total = db.query(GalleryComment).filter(GalleryComment.image_id == image.id).count()

    offset = (page - 1) * per_page
    comments = (
        db.query(GalleryComment)
        .filter(GalleryComment.image_id == image.id)
        .order_by(GalleryComment.created_at.desc())
        .offset(offset)
        .limit(per_page)
        .all()
    )

    return CommentListResponse(
        total=total,
        comments=[
            CommentResponse(
                id=c.id,
                display_name=c.display_name,
                text=c.text,
                created_at=c.created_at.isoformat(),
            )
            for c in comments
        ],
    )


@router.post("/image/{share_id}/share", response_model=ShareResponse)
@limiter.limit("30/minute")
async def track_share(
    request: Request,
    share_id: str,
    share_req: ShareRequest,
    db: DbSession,
):
    """Track when an image is shared to a platform."""
    image = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.share_id == share_id,
            GeneratedImage.is_public.is_(True),  # noqa: E712
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    new_share = GalleryShare(
        image_id=image.id,
        platform=share_req.platform,
    )
    db.add(new_share)
    image.share_count = (image.share_count or 0) + 1
    db.commit()

    # Generate platform-specific share URL
    base_url = str(request.base_url).rstrip("/")
    share_url = f"{base_url}/share/{share_id}"

    return ShareResponse(
        success=True,
        share_count=image.share_count or 0,
        share_url=share_url,
    )


@router.post("/publish", response_model=PublishResponse)
@limiter.limit("10/minute")
async def publish_to_gallery(
    request: Request,
    publish_req: PublishRequest,
    db: DbSession,
):
    """Publish a private image to the public gallery."""
    image = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.id == publish_req.image_id)
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    if image.is_public:
        raise HTTPException(status_code=400, detail="Image is already public")

    # Generate share ID and make public
    image.share_id = generate_share_id()
    image.is_public = True
    db.commit()

    base_url = str(request.base_url).rstrip("/")
    share_url = f"{base_url}/share/{image.share_id}"

    return PublishResponse(
        success=True,
        share_id=image.share_id,
        share_url=share_url,
    )


@router.delete("/image/{share_id}")
@limiter.limit("10/minute")
async def unpublish_from_gallery(
    request: Request,
    share_id: str,
    db: DbSession,
):
    """Remove an image from the public gallery (make private)."""
    image = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.share_id == share_id,
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    image.is_public = False
    db.commit()

    return {"success": True, "message": "Image removed from public gallery"}


@router.get("/stats")
@limiter.limit("30/minute")
async def get_gallery_stats(
    request: Request,
    db: DbSession,
):
    """Get community gallery statistics."""
    total_public = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.is_public.is_(True))  # noqa: E712
        .count()
    )

    total_likes = (
        db.query(func.sum(GeneratedImage.like_count))
        .filter(GeneratedImage.is_public.is_(True))  # noqa: E712
        .scalar()
        or 0
    )

    total_comments = db.query(GalleryComment).count()

    total_shares = db.query(GalleryShare).count()

    # Most popular tags
    # Note: This is a simplified query; for production, consider caching
    top_images = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.is_public.is_(True))  # noqa: E712
        .order_by(GeneratedImage.like_count.desc())
        .limit(10)
        .all()
    )

    return {
        "total_public_images": total_public,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "total_shares": total_shares,
        "trending": [image_to_response(img) for img in top_images[:5]],
    }


# ========== Collection Models ==========


class CollectionCreate(BaseModel):
    """Request to create a collection."""

    name: str
    description: str | None = None
    theme: str | None = None
    artwork_ids: list[int] = []


class CollectionResponse(BaseModel):
    """Collection details response."""

    id: int
    name: str
    description: str | None
    theme: str | None
    cover_image_url: str | None
    artwork_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class CollectionDetailResponse(CollectionResponse):
    """Collection with artworks."""

    artworks: list[dict]


# ========== Collection Endpoints ==========


@router.get("/collections", response_model=list[CollectionResponse])
@limiter.limit("60/minute")
async def list_collections(
    request: Request,
    db: DbSession,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[CollectionResponse]:
    """List all public collections."""
    collections = (
        db.query(GalleryCollection)
        .filter(GalleryCollection.is_public.is_(True))  # noqa: E712
        .order_by(GalleryCollection.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    results = []
    for coll in collections:
        artwork_count = (
            db.query(CollectionArtwork)
            .filter(CollectionArtwork.collection_id == coll.id)
            .count()
        )

        cover_url = None
        if coll.cover_image:
            cover_url = f"/gallery/{coll.cover_image.filename}"

        results.append(
            CollectionResponse(
                id=coll.id,
                name=coll.name,
                description=coll.description,
                theme=coll.theme,
                cover_image_url=cover_url,
                artwork_count=artwork_count,
                created_at=coll.created_at,
            )
        )

    return results


@router.post("/collections", response_model=CollectionResponse)
@limiter.limit("10/minute")
async def create_collection(
    request: Request,
    collection_req: CollectionCreate,
    db: DbSession,
) -> CollectionResponse:
    """Create a new collection."""
    # Create collection
    collection = GalleryCollection(
        name=collection_req.name,
        description=collection_req.description,
        theme=collection_req.theme,
        created_by_aria=True,
    )
    db.add(collection)
    db.flush()  # Get ID

    # Add artworks
    for i, img_id in enumerate(collection_req.artwork_ids):
        # Verify image exists
        if db.query(GeneratedImage).filter(GeneratedImage.id == img_id).first():
            artwork = CollectionArtwork(
                collection_id=collection.id,
                image_id=img_id,
                position=i,
            )
            db.add(artwork)

            # Set first image as cover if not set
            if i == 0 and not collection.cover_image_id:
                collection.cover_image_id = img_id

    db.commit()
    db.refresh(collection)

    cover_url = None
    if collection.cover_image:
        cover_url = f"/gallery/{collection.cover_image.filename}"

    return CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        theme=collection.theme,
        cover_image_url=cover_url,
        artwork_count=len(collection_req.artwork_ids),
        created_at=collection.created_at,
    )


@router.get("/collections/{collection_id}", response_model=CollectionDetailResponse)
@limiter.limit("60/minute")
async def get_collection(
    request: Request,
    collection_id: int,
    db: DbSession,
) -> CollectionDetailResponse:
    """Get collection details with artworks."""
    collection = (
        db.query(GalleryCollection)
        .filter(GalleryCollection.id == collection_id)
        .first()
    )

    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Get artworks in order
    artworks_query = (
        db.query(CollectionArtwork)
        .filter(CollectionArtwork.collection_id == collection_id)
        .order_by(CollectionArtwork.position)
        .all()
    )

    artworks = []
    for ca in artworks_query:
        img = ca.image
        artworks.append(
            {
                "id": img.id,
                "filename": img.filename,
                "image_url": f"/gallery/{img.filename}",
                "prompt": img.prompt,
                "mood": (
                    img.generation_params.get("mood") if img.generation_params else None
                ),
                "position": ca.position,
            }
        )

    cover_url = None
    if collection.cover_image:
        cover_url = f"/gallery/{collection.cover_image.filename}"

    return CollectionDetailResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        theme=collection.theme,
        cover_image_url=cover_url,
        artwork_count=len(artworks),
        created_at=collection.created_at,
        artworks=artworks,
    )


# ========== Advanced Search ==========


class SearchFilters(BaseModel):
    """Search filter parameters."""

    query: str | None = None
    mood: str | None = None
    min_score: float | None = None
    max_score: float | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    has_likes: bool | None = None
    sort_by: str = "newest"  # newest, oldest, best, most_liked, trending


class SearchResponse(BaseModel):
    """Search results response."""

    results: list[dict]
    total: int
    filters_applied: dict


@router.post("/search", response_model=SearchResponse)
@limiter.limit("60/minute")
async def advanced_search(
    request: Request,
    filters: SearchFilters,
    db: DbSession,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
) -> SearchResponse:
    """Advanced search with multiple filters."""
    query = db.query(GeneratedImage).filter(
        GeneratedImage.is_public.is_(True)  # noqa: E712
    )

    filters_applied: dict = {}

    # Text search
    if filters.query:
        search_term = f"%{filters.query}%"
        query = query.filter(
            or_(
                GeneratedImage.prompt.ilike(search_term),
                GeneratedImage.tags.cast(str).ilike(search_term),
            )
        )
        filters_applied["query"] = filters.query

    # Mood filter (search in generation_params JSON)
    if filters.mood:
        # This is a simplification - actual implementation depends on DB JSON support
        query = query.filter(
            GeneratedImage.generation_params["mood"].astext == filters.mood
        )
        filters_applied["mood"] = filters.mood

    # Score filter
    if filters.min_score is not None:
        query = query.filter(GeneratedImage.final_score >= filters.min_score)
        filters_applied["min_score"] = filters.min_score

    if filters.max_score is not None:
        query = query.filter(GeneratedImage.final_score <= filters.max_score)
        filters_applied["max_score"] = filters.max_score

    # Date filter
    if filters.date_from:
        query = query.filter(GeneratedImage.created_at >= filters.date_from)
        filters_applied["date_from"] = filters.date_from.isoformat()

    if filters.date_to:
        query = query.filter(GeneratedImage.created_at <= filters.date_to)
        filters_applied["date_to"] = filters.date_to.isoformat()

    # Has likes filter
    if filters.has_likes is not None:
        if filters.has_likes:
            query = query.filter(GeneratedImage.like_count > 0)
        else:
            query = query.filter(GeneratedImage.like_count == 0)
        filters_applied["has_likes"] = filters.has_likes

    # Get total before pagination
    total = query.count()

    # Sorting
    sort_mapping = {
        "newest": GeneratedImage.created_at.desc(),
        "oldest": GeneratedImage.created_at.asc(),
        "best": GeneratedImage.final_score.desc().nullslast(),
        "most_liked": GeneratedImage.like_count.desc(),
        "trending": GeneratedImage.view_count.desc(),  # Simple trending
    }

    order = sort_mapping.get(filters.sort_by, GeneratedImage.created_at.desc())
    query = query.order_by(order)

    # Pagination
    results = query.offset(offset).limit(limit).all()

    # Format results
    formatted = []
    for img in results:
        formatted.append(
            {
                "id": img.id,
                "filename": img.filename,
                "image_url": f"/gallery/{img.filename}",
                "prompt": img.prompt,
                "score": img.final_score,
                "like_count": img.like_count,
                "view_count": img.view_count,
                "created_at": img.created_at.isoformat() if img.created_at else None,
            }
        )

    return SearchResponse(
        results=formatted,
        total=total,
        filters_applied=filters_applied,
    )


# ========== Trending Endpoint ==========


@router.get("/trending")
@limiter.limit("60/minute")
async def get_trending(
    request: Request,
    db: DbSession,
    time_window: str = Query(default="week", pattern="^(day|week|month|all)$"),
    limit: int = Query(default=20, le=50),
) -> dict:
    """Get trending artworks based on engagement velocity."""
    # Calculate time window
    now = datetime.utcnow()
    windows = {
        "day": timedelta(days=1),
        "week": timedelta(weeks=1),
        "month": timedelta(days=30),
        "all": timedelta(days=365 * 10),  # Effectively all time
    }
    cutoff = now - windows.get(time_window, timedelta(weeks=1))

    # Get images with recent engagement
    # Simple trending: weight recent likes more heavily
    trending_query = (
        db.query(GeneratedImage, func.count(GalleryLike.id).label("recent_likes"))
        .outerjoin(
            GalleryLike,
            and_(
                GalleryLike.image_id == GeneratedImage.id,
                GalleryLike.created_at >= cutoff,
            ),
        )
        .filter(GeneratedImage.is_public.is_(True))  # noqa: E712
        .group_by(GeneratedImage.id)
        .order_by(
            func.count(GalleryLike.id).desc(),
            GeneratedImage.view_count.desc(),
        )
        .limit(limit)
        .all()
    )

    results = []
    for img, recent_likes in trending_query:
        results.append(
            {
                "id": img.id,
                "filename": img.filename,
                "image_url": f"/gallery/{img.filename}",
                "prompt": img.prompt,
                "like_count": img.like_count,
                "recent_likes": recent_likes,
                "view_count": img.view_count,
                "trending_score": recent_likes * 2 + (img.view_count or 0) * 0.1,
            }
        )

    return {
        "time_window": time_window,
        "trending": results,
    }
