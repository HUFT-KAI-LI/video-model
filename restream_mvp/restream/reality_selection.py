"""Selection-policy identity and provenance of reference-derived assets.

Reference sets must be a deterministic function of the declared selection
configuration. Any change to the fields below invalidates every manifest, cached
pool and derived control asset (notably the train-split global mean), so the
builder, the feature-cache script, the Dataset and the global-mean loader all
compute the same hash from the same functions.

The identity deliberately separates:
  * ``data.selection_seed``   -- manifest construction RNG, independent of the
    training/probing ``seed``, so a training seed change does not force a data
    rebuild and a data seed change cannot silently keep the old hash;
  * ``data.temporal_sampling``-- the frame-selection policy shared by target
    decoding, the target histogram and reference selection. Offline manifests
    use ``first_at_or_after``; strict-online manifests use ``causal_previous``
    (last frame at or before each requested timestamp);
  * ``shot_filter_hash``      -- the semantic shot-filter fields that decide
    which shots are eligible, not compute knobs such as worker counts;
  * ``pool_size``             -- the explicit reference-pool size, no longer
    inferred from the evaluation sweep.

Bump SELECTION_IDENTITY_SCHEMA whenever the identity fields or hash semantics
change. GLOBAL_MEAN_SCHEMA identifies the payload layout written by
``scripts/cache_reality_features.py`` and read by ``reality_paired``.
"""
import copy
import hashlib
import random

from .reality_data import canonical_hash

SELECTION_IDENTITY_SCHEMA = 3
# Schema 2 adds role-matched control means (async / aligned / positive).
GLOBAL_MEAN_SCHEMA = 2
GLOBAL_MEAN_ROLES = {"async": ("async",), "aligned": ("aligned",), "positive": ("async", "aligned")}
TEMPORAL_SAMPLING = ("first_at_or_after", "causal_previous")
PROTOCOL_TEMPORAL_SAMPLING = {"offline_target_filtered": "first_at_or_after",
                              "strict_online": "causal_previous"}
SHOT_FILTER_FIELDS = ("histogram_cut", "pixel_jump", "black_level", "black_fraction")


def validate_temporal_sampling(protocol, temporal_sampling):
    """The sampling policy is fixed by the protocol, not a free knob."""
    if temporal_sampling not in TEMPORAL_SAMPLING:
        raise ValueError(f"Invalid temporal_sampling {temporal_sampling!r}")
    required = PROTOCOL_TEMPORAL_SAMPLING.get(protocol)
    if required is None:
        raise ValueError("Invalid reference selection protocol")
    if temporal_sampling != required:
        raise ValueError(f"{protocol} requires temporal_sampling={required}, got {temporal_sampling}")


def shot_filter_identity(config):
    filter_config = config["reality_memory"]["filter"]
    missing = [name for name in SHOT_FILTER_FIELDS if name not in filter_config]
    if missing:
        raise ValueError(f"Shot filter configuration is missing {missing}")
    return {name: filter_config[name] for name in SHOT_FILTER_FIELDS}


def selection_identity(config):
    """Declared rule that produced a manifest's reference_sets and derived assets.

    Fields that only decide which training rows exist (mixture probabilities,
    per-sample K) are deliberately excluded: they do not change which references
    are available to a row, so they must not invalidate cached pools or the
    global mean. Fields that change availability or sampling rules are included.
    """
    memory = config["reality_memory"]
    refs = memory["references"]
    data = config["data"]
    for name, value in (("data.selection_seed", data.get("selection_seed")),
                        ("data.temporal_sampling", data.get("temporal_sampling")),
                        ("references.pool_size", refs.get("pool_size"))):
        if value is None:
            raise ValueError(f"Selection identity requires explicit {name}; rebuild config/manifest")
    protocol = refs.get("selection_protocol", "offline_target_filtered")
    validate_temporal_sampling(protocol, data["temporal_sampling"])
    return {
        "schema": SELECTION_IDENTITY_SCHEMA,
        "protocol": protocol,
        "async_direction": refs.get("async_direction", "past_only"),
        "min_gap_sec": refs["min_gap_sec"],
        "near_radius_sec": refs["near_radius_sec"],
        "boundary_margin_sec": refs["boundary_margin_sec"],
        "prefix_latents": memory["objective"]["prefix_latents"],
        "frames": data["frames"],
        "fps": data["fps"],
        "scene_similarity": memory["filter"]["scene_similarity"],
        "pool_size": refs["pool_size"],
        "min_count": refs["min_count"],
        "selection_seed": data["selection_seed"],
        "temporal_sampling": data["temporal_sampling"],
        "shot_filter_hash": canonical_hash(shot_filter_identity(config)),
    }


def selection_config_hash(config):
    """One hash for the builder, the Dataset and every derived control asset."""
    return canonical_hash(selection_identity(config))


def normalize_selection_defaults(config):
    """Fill pre-identity defaults so older configs (e.g. a saved checkpoint) can
    be compared and hashed with their *effective* semantics.

    Old paired configs predate ``selection_protocol``, ``selection_seed``,
    ``temporal_sampling``, ``pool_size`` and the diagnostic flags; the values
    filled here are exactly the ones those runs behaved with.
    """
    config = copy.deepcopy(config)
    data = config.setdefault("data", {})
    memory = config.setdefault("reality_memory", {})
    refs = memory.setdefault("references", {})
    protocol = refs.setdefault("selection_protocol", "offline_target_filtered")
    data.setdefault("selection_seed", config.get("seed"))
    data.setdefault("temporal_sampling", PROTOCOL_TEMPORAL_SAMPLING.get(protocol))
    if refs.get("pool_size") is None:
        counts = (config.get("eval") or {}).get("reference_counts") or [0]
        refs["pool_size"] = max(int(refs.get("max_count", 0)), *(int(count) for count in counts))
    refs.setdefault("allow_legacy_offline_manifest", False)
    refs.setdefault("global_constant_features", "data/reality_global_mean_features.pt")
    paired = (memory.get("objective") or {}).get("paired")
    if isinstance(paired, dict):
        paired.setdefault("diagnostic_gradients", True)
        paired.setdefault("diagnostic_interval", 10)
    return config


def validate_selection_protocol(protocol, async_direction):
    if protocol not in PROTOCOL_TEMPORAL_SAMPLING:
        raise ValueError("Invalid reference selection protocol")
    if protocol == "strict_online" and async_direction != "past_only":
        raise ValueError("strict_online requires past_only references")


def unique_train_reference_pool(rows, cache, kinds=("async", "aligned")):
    """Deduplicated union of the positive reference sets of every train row.

    The global-mean control must be insensitive to training mixture details:
    no_memory/wrong probabilities, the per-sample random K and donor references
    duplicated across rows must not double-count a single cached frame. It must
    also be provably train-only: every row and every contributing reference is
    checked for split, source and shot agreement instead of trusting the file.
    """
    unique = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"Global-mean pool received a non-train row: {row.get('sample_id')} split={row.get('split')}")
        for kind in kinds:
            for ref in row.get("reference_sets", {}).get(kind, ()):
                if ref.get("split") != "train":
                    raise ValueError(f"Global-mean pool saw a non-train {kind} reference in {row.get('sample_id')}")
                if ref.get("source_id") != row.get("source_id"):
                    raise ValueError(f"Global-mean pool saw a foreign-source {kind} reference in {row.get('sample_id')}")
                if ref.get("shot_id") != row.get("shot_id"):
                    raise ValueError(f"Global-mean pool saw a cross-shot {kind} reference in {row.get('sample_id')}")
                unique[cache.key(ref)] = ref
    return unique


def role_reference_pool(rows, cache, role):
    """Deduplicated train pool for one global-control role.

    ``async``/``aligned`` are role-matched to the corresponding correct kind;
    ``positive`` is their union (the original scene-independent control).
    """
    if role not in GLOBAL_MEAN_ROLES:
        raise ValueError(f"Unknown global-mean role {role!r}; expected one of {sorted(GLOBAL_MEAN_ROLES)}")
    return unique_train_reference_pool(rows, cache, kinds=GLOBAL_MEAN_ROLES[role])


def select_targets(rows, count, seed):
    """Deterministic unique-target index selection.

    ``count <= 0`` or ``count >= len(rows)`` selects every target; otherwise a
    seeded sample without replacement is sorted for a stable, persisted order.
    """
    if count is None or count <= 0 or count >= len(rows):
        return list(range(len(rows)))
    return sorted(random.Random(seed).sample(range(len(rows)), count))


def manifest_digest(path):
    """SHA-256 of a manifest file as stored on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sorted_reference_keys(unique):
    return sorted(unique)


def reference_keys_digest(unique):
    """Deterministic digest over the deduplicated, sorted content keys."""
    return canonical_hash(sorted_reference_keys(unique))
