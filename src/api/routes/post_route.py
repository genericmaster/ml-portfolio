from fastapi import APIRouter, HTTPException
from pathlib import Path
import markdown
import frontmatter

writing_router = APIRouter()

POSTS_DIR = Path("posts")

@writing_router.get("/writing/{slug}")
def get_post(slug: str):
    post_path = POSTS_DIR / f"{slug}.md"
    if not post_path.exists():
        raise HTTPException(status_code=404, detail="Post not found")
    post = frontmatter.load(str(post_path))
    html = markdown.markdown(
        post.content,
        extensions=["fenced_code", "codehilite", "tables", "nl2br"]
    )
    return {
        "html": html,
        "title": post.metadata.get("title", slug),
        "date": str(post.metadata.get("date", "")),
        "tag": post.metadata.get("tag", "")
    }

@writing_router.get("/writing")
def list_posts():
    if not POSTS_DIR.exists():
        return {"posts": []}
    posts = []
    for path in sorted(POSTS_DIR.glob("*.md"), reverse=True):
        post = frontmatter.load(str(path))
        posts.append({
            "slug": path.stem,
            "title": post.metadata.get("title", path.stem),
            "date": str(post.metadata.get("date", "")),
            "tag": post.metadata.get("tag", "")
        })
    return {"posts": posts}