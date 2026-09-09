"""Selection-policy identity shared by the manifest builder, the feature-cache
preparation script and Dataset load-time checks.

Reference sets are meant to be a deterministic function of the declared
selection configuration alone. Any change to the fields below invalidates every
manifest, cached pool and derived control asset (notably the train-split global
mean) built under the old policy, so builder and loader must compute the same
hash from the same function.

Bump SELECTION_IDENTITY_SCHEMA whenever the identity fields or hash semantics
change. GLOBAL_MEAN_SCHEMA identifies the payload layout written by
``scripts/cache_reality_features.py`` and read by ``reality_paired``; a stale
global-constant file (missing schema, wrong manifest digest or selection hash)
is refused at load time instead of silently pairing with current data.
"""
import hashlib

from .reality_data import canonical_hash

SELECTION_IDENTITY_SCHEMA = 2
GLOBAL_MEAN_SCHEMA = 1


def selection_identity(config):
    """Declared rule that produced a manifest's reference_sets and derived assets.

    Fields that only decide which training rows exist (mixture probabilities,
    per-sample K) are deliberately excluded: they do not change which references
    are available to a row, so they must not invalidate cached pools or the
    global mean. Fields that change the availability rules are included.
    """
    memory = config["reality_memory"]
    refs = memory["references"]
    data = config["data"]
    eval_counts = (config.get("eval") or {}).get("reference_counts") or [0]
    pool_size = max(int(refs["max_count"]), *(int(count) for count in eval_counts))
    return {
        "schema": SELECTION_IDENTITY_SCHEMA,
        "protocol": refs.get("selection_protocol", "offline_target_filtered"),
        "async_direction": refs.get("async_direction", "past_only"),
        "min_gap_sec": refs["min_gap_sec"],
        "near_radius_sec": refs["near_radius_sec"],
        "boundary_margin_sec": refs["boundary_margin_sec"],
        "prefix_latents": memory["objective"]["prefix_latents"],
        "frames": data["frames"],
        "fps": data["fps"],
        "scene_similarity": memory["filter"]["scene_similarity"],
        "pool_size": pool_size,
        "min_count": refs["min_count"],
    }


def selection_config_hash(config):
    """One hash for both the manifest builder and Dataset/global-mean loaders."""
    return canonical_hash(selection_identity(config))


def validate_selection_protocol(protocol, async_direction):
    if protocol not in ("offline_target_filtered", "strict_online"):
        raise ValueError("Invalid reference selection protocol")
    if protocol == "strict_online" and async_direction != "past_only":
        raise ValueError("strict_online requires past_only references")


def unique_train_reference_pool(rows, cache, kinds=("async", "aligned")):
    """Deduplicated union of the positive reference sets of every manifest row.

    The global-mean control must be insensitive to training mixture details:
    no_memory/wrong probabilities, the per-sample random K and donor references
    duplicated across rows must not double-count a single cached frame. The
    result maps a content key (video frame + encoder identity) to one reference.
    """
    unique = {}
    for row in rows:
        for kind in kinds:
            for ref in row.get("reference_sets", {}).get(kind, ()):
                unique[cache.key(ref)] = ref
    return unique


def manifest_digest(path):
    """SHA-256 of a manifest file as stored on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sorted_reference_keys(unique):
    return sorted(unique)


def reference_keys_digest(unique):
    """Deterministic digest over the deduplicated, sorted content keys."""
    return canonical_hash(sorted_reference_keys(unique))
