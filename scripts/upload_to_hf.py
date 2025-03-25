from huggingface_hub import HfApi
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)
import re
import semver


def get_latest_version(repo_id):
    """Get the latest semantic version tag from a repository"""
    api = HfApi()

    # Get all repo refs (tags and branches)
    refs = api.list_repo_refs(repo_id)

    # Extract just the tag names from the GitRefs object
    tags = [tag.ref.replace("refs/tags/", "") for tag in refs.tags]

    # Filter for semantic version tags (vX.Y.Z)
    version_tags = [tag for tag in tags if re.match(r"^v\d+\.\d+\.\d+$", tag)]

    if not version_tags:
        return None

    # Sort tags by semantic version
    sorted_tags = sorted(
        version_tags,
        key=lambda tag: semver.Version.parse(tag.lstrip("v")),
        reverse=True,
    )

    return sorted_tags[0] if sorted_tags else None


def increment_version(version, increment_type="patch"):
    """Increment a semantic version string"""
    if version is None:
        return "v0.1.0"  # Initial version

    # Remove 'v' prefix if present
    version_str = version.lstrip("v")
    version_obj = semver.Version.parse(version_str)

    if increment_type == "major":
        new_version = version_obj.bump_major()
    elif increment_type == "minor":
        new_version = version_obj.bump_minor()
    else:  # patch
        new_version = version_obj.bump_patch()

    return f"v{new_version}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--version", help="Explicit version tag (e.g., v1.0.0)")
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Type of version increment if --version not specified",
    )
    args = parser.parse_args()

    # Load and push the model
    checkpoint = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )

    # Push model to appropriate repository
    checkpoint.model.push_to_hub(repo_id=args.repo)

    # Handle versioning
    version_tag = args.version
    if not version_tag:
        latest_version = get_latest_version(args.repo)
        version_tag = increment_version(latest_version, args.bump)

    # Create the tag
    api = HfApi()
    api.create_tag(
        repo_id=args.repo,
        tag=version_tag,
    )
    print(f"Created tag {version_tag} for repository {args.repo}")
