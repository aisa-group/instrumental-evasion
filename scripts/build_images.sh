#!/usr/bin/env bash
# Build every Apptainer SIF the 50 tasks need into $IE_IMAGE_DIR.
#
#   scripts/build_images.sh [--family base|tb2|tblite|all] [--force]
#                           [--tb2-repo DIR] [--tblite-repo DIR]
#
# The 50 tasks resolve their images through instrumental_evasion.tasks.tb_spec:
# a per-task env var, else $IE_IMAGE_DIR/<basename>.sif (IE_IMAGE_DIR defaults to
# ~/.cache/instrumental-evasion/images). Run `python -m
# instrumental_evasion.tasks.registry` to list each task's resolved path. This
# script writes exactly those basenames.
#
# Seventeen images in three families:
#   base    tb-agent-stateful.sif  gitpush.sif  tb-db-entry.sif
#   tb2     nine  tb2-<task>.sif    (Terminal-Bench 2 tasks)
#   tblite  five  tb-<task>.sif     (OpenThoughts-TBLite tasks)
#
# Prerequisites
#   * apptainer on PATH (unprivileged is fine; no root or fakeroot required).
#   * A local scratch directory on a LOCAL disk for the Apptainer cache and
#     tmpdir. The default is /tmp; override with IE_BUILD_SCRATCH. Never point it
#     at a networked filesystem: the Apptainer cache takes a file lock that can
#     hang forever there, and unpacking many small files stalls a distributed
#     metadata server. The finished SIF is one large file, so
#     publishing it to a shared $IE_IMAGE_DIR is a single bulk copy.
#   * For the tb2 family: a local clone of
#         https://github.com/harbor-framework/terminal-bench-2
#     checked out at commit $TB2_COMMIT, passed with --tb2-repo or $TB2_REPO.
#     The tb2 build pulls each task's prebuilt upstream docker image (named in
#     that task's task.toml) and stages the vendored verifier into it.
#   * For the tblite family: a local clone of
#         https://github.com/open-thoughts/OpenThoughts-TBLite
#     checked out at commit $TBLITE_COMMIT, passed with --tblite-repo or
#     $TBLITE_REPO. The tblite build replays each task's upstream Dockerfile.
#   * Building base images from the images/*.Dockerfile files uses a Docker
#     daemon when one is reachable; otherwise it falls back to an
#     Apptainer-native build that reproduces the same rootfs. No Docker is
#     required either way.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

# Pinned upstream commits. The build refuses a clone at any other commit so a
# published image can always be traced to an exact task definition.
TB2_COMMIT="2fd12b88aafdd04a52c298e3940bcb189f9766d6"
TBLITE_COMMIT="5c37b41f00ce04719a4453061076ae9f46b74b7d"

OUT="${IE_IMAGE_DIR:-$HOME/.cache/instrumental-evasion/images}"
OUT="${OUT/#\~/$HOME}"
FAMILY="all"
FORCE=0
TB2_REPO="${TB2_REPO:-}"
TBLITE_REPO="${TBLITE_REPO:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family) FAMILY="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --tb2-repo) TB2_REPO="$2"; shift 2 ;;
        --tblite-repo) TBLITE_REPO="$2"; shift 2 ;;
        -h|--help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$FAMILY" in base|tb2|tblite|all) ;; *) echo "bad --family: $FAMILY" >&2; exit 2 ;; esac

command -v apptainer >/dev/null 2>&1 || { echo "apptainer is not on PATH" >&2; exit 1; }

# Local scratch for the Apptainer cache/tmpdir. Must be local disk (see above).
SCRATCH="${IE_BUILD_SCRATCH:-/tmp}"
if [[ ! -d "$SCRATCH" || ! -w "$SCRATCH" ]]; then
    echo "build scratch directory is not writable: $SCRATCH (set IE_BUILD_SCRATCH)" >&2
    exit 1
fi
mkdir -p "$OUT"
echo "images -> $OUT"
echo "scratch -> $SCRATCH"

# Fresh local scratch per invocation, cleaned on exit; every path Apptainer
# touches stays on local disk.
WORK="$(mktemp -d "$SCRATCH/ie-images-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export APPTAINER_CACHEDIR="$WORK/cache" APPTAINER_TMPDIR="$WORK/tmp" TMPDIR="$WORK/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$TMPDIR"

# Publish a staged SIF to its final path atomically: copy to a temp name and
# rename, so a reader never sees a partial image and a failed copy cannot leave
# a broken container in place.
publish() {
    local staged="$1" target="$2"
    mkdir -p "$(dirname "$target")"
    cp "$staged" "$target.tmp.$$"
    mv -f "$target.tmp.$$" "$target"
    echo "published $target"
}

docker_ok() { timeout 20 docker info >/dev/null 2>&1; }

# --- base images ----------------------------------------------------------
#
# Each base image is defined by images/<name>.Dockerfile. When a Docker daemon
# is reachable the canonical path is docker build -> docker save -> apptainer
# build docker-archive://. Without Docker, build_base_native reproduces the same
# rootfs unprivileged: %post cannot run (it needs to mount /proc, which is not
# permitted unprivileged), so package installs run through `exec --writable` on
# an intermediate sandbox directory instead.

# build_base_docker <dockerfile-basename> <staged-sif>
build_base_docker() {
    local name="$1" staged="$2"
    local tag="ie-$name:build-$$"
    echo "==> docker build images/$name.Dockerfile"
    docker build -t "$tag" -f "images/$name.Dockerfile" images
    docker save "$tag" -o "$WORK/$name.tar"
    apptainer build "$staged" "docker-archive://$WORK/$name.tar"
    docker rmi "$tag" >/dev/null 2>&1 || true
}

# apt-get install needs root, which we do not have unprivileged, so the packages
# are downloaded WITH their dependency closure and unpacked with dpkg-deb -x.
# Only packages the base does NOT already have are unpacked: laying libc6/dpkg/
# tar over a live rootfs breaks it (the running dpkg-deb loses its own loader).
NATIVE_APT_INSTALL='
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    tmp=$(mktemp -d) && cd "$tmp"
    closure=$(apt-cache depends --recurse --no-recommends --no-suggests \
        --no-conflicts --no-breaks --no-replaces --no-enhances "$@" \
        | grep "^[[:alnum:]]" | sort -u)
    apt-get download $closure 2>/dev/null || true
    for d in *.deb; do
        pkg=$(dpkg-deb -f "$d" Package)
        if dpkg-query -s "$pkg" 2>/dev/null | grep -q "^Status: install ok installed"; then
            continue
        fi
        dpkg-deb -x "$d" /
    done
    cd / && rm -rf "$tmp" /var/lib/apt/lists/*
'

# build_base_native <base-docker-ref> <staged-sif> [apt pkgs...] -- with a
# trailing "PIP:<spec>" arg meaning pip install that spec, and "MKDIR:<path>".
build_base_native() {
    local base="$1" staged="$2"; shift 2
    local sandbox="$WORK/rootfs-$RANDOM"
    echo "==> apptainer native build from docker://$base"
    apptainer build --sandbox "$sandbox" "docker://$base"
    local apt=()
    for step in "$@"; do
        case "$step" in
            PIP:*)
                echo "==> pip install ${step#PIP:} (no %post, so --writable)"
                apptainer exec --writable "$sandbox" \
                    python -m pip install --no-cache-dir "${step#PIP:}" ;;
            MKDIR:*)
                mkdir -p "$sandbox/${step#MKDIR:}" ;;
            *) apt+=("$step") ;;
        esac
    done
    if [[ ${#apt[@]} -gt 0 ]]; then
        echo "==> installing apt packages unprivileged: ${apt[*]}"
        apptainer exec --writable \
            --env http_proxy="${http_proxy:-}" --env https_proxy="${https_proxy:-}" \
            "$sandbox" bash -lc "$NATIVE_APT_INSTALL" _ "${apt[@]}"
    fi
    echo "==> packing the sandbox into a SIF"
    apptainer build "$staged" "$sandbox"
    rm -rf "$sandbox"
}

build_agent_stateful() {
    local target="$OUT/tb-agent-stateful.sif"
    if [[ -s "$target" && $FORCE -eq 0 ]]; then echo "SKIP tb-agent-stateful (exists)"; return; fi
    local staged="$WORK/agent-stateful.sif"
    if docker_ok; then build_base_docker agent-stateful "$staged"
    else build_base_native python:3.12-slim "$staged"; fi
    echo "==> verifying before publishing"
    apptainer exec --contain --no-home --cleanenv "$staged" python --version
    publish "$staged" "$target"
}

build_gitpush() {
    local target="$OUT/gitpush.sif"
    if [[ -s "$target" && $FORCE -eq 0 ]]; then echo "SKIP gitpush (exists)"; return; fi
    local staged="$WORK/gitpush.sif"
    if docker_ok; then build_base_docker git "$staged"
    else build_base_native python:3.12.13-bookworm "$staged" "PIP:pytest==9.1.1"; fi
    echo "==> verifying the toolchain before publishing"
    apptainer exec --contain --no-home --cleanenv "$staged" bash -lc '
        set -e
        git --version
        make --version | head -1
        python --version
        python -m pytest --version
    '
    publish "$staged" "$target"
}

build_db_entry() {
    local target="$OUT/tb-db-entry.sif"
    if [[ -s "$target" && $FORCE -eq 0 ]]; then echo "SKIP tb-db-entry (exists)"; return; fi
    local staged="$WORK/db-entry.sif"
    if docker_ok; then build_base_docker db-entry "$staged"
    else build_base_native python:3.12-slim-bookworm "$staged" sqlite3 ca-certificates MKDIR:app/data; fi
    echo "==> verifying the image before publishing"
    apptainer exec --contain --no-home --cleanenv "$staged" bash -lc '
        set -e
        python3 --version
        python3 -c "import sqlite3; sqlite3.connect(\":memory:\").execute(\"create table t(x)\")"
        sqlite3 --version >/dev/null || { echo "sqlite3 shell missing" >&2; exit 1; }
        test -d /app/data
        # The task state is seeded by the harness; a database in the image would
        # mean two starting states, one of them stale.
        if compgen -G "/app/data/*" >/dev/null; then
            echo "FATAL: /app/data is not empty in the image" >&2; exit 1
        fi
    '
    publish "$staged" "$target"
}

# --- upstream clone validation --------------------------------------------

# require_clone <path> <expected-commit> <name> <url>
require_clone() {
    local path="$1" want="$2" name="$3" url="$4"
    [[ -n "$path" ]] || { echo "$name clone not given (--${name}-repo or \$${name^^}_REPO); clone $url" >&2; exit 1; }
    [[ -d "$path/.git" ]] || { echo "$path is not a git clone of $url" >&2; exit 1; }
    local have
    have="$(git -C "$path" rev-parse HEAD)"
    if [[ "$have" != "$want" ]]; then
        echo "$name clone is at $have but the pinned commit is $want" >&2
        echo "  run: git -C $path checkout $want" >&2
        exit 1
    fi
    echo "$name clone OK at $want"
}

# --- Terminal-Bench 2 ------------------------------------------------------
#
# The paper's nine TB-2 tasks. The apt/run recipes are what the task's REFERENCE
# solution installs or fetches at runtime; upstream can do that because
# terminal-bench grades with internet, while the eval sandbox here is sealed and
# its rootfs read-only, so anything the honest route needs must be baked in
# first. A task absent from these maps needs nothing beyond its base image.
TB2_TASKS=(bn-fit-modify build-pmars cobol-modernization constraints-scheduling
           distribution-search fix-code-vulnerability log-summary-date-ranges
           multi-source-data-merger sqlite-db-truncate)

declare -A TB2_APT=(
    [build-pmars]="build-essential libncurses-dev dpkg-dev"
    [cobol-modernization]="gnucobol3"
    [bn-fit-modify]="r-base-dev"
)
declare -A TB2_RUN=(
    # bnlearn 5.x needs R >= 4.4; the base ships R 4.3, so pin the version the
    # upstream reference solution itself installs. Unpinned, install.packages()
    # only warns and exits 0, leaving an image that looks built and cannot solve.
    [bn-fit-modify]="R --slave -e 'install.packages(\"remotes\", repos=\"https://cloud.r-project.org\")' && R --slave -e 'remotes::install_version(\"bnlearn\", version=\"4.9\", repos=\"https://cloud.r-project.org\", upgrade=\"never\")' && R --slave -e 'library(bnlearn); cat(\"bnlearn ok\n\")'"
    # pmars is built from the Debian source package, not the binary.
    [build-pmars]="echo 'deb-src http://deb.debian.org/debian stable main' > /etc/apt/sources.list.d/deb-src.list && apt-get update && cd /app && apt-get source pmars && ls -d /app/pmars-*/debian"
)

build_tb2() {
    require_clone "$TB2_REPO" "$TB2_COMMIT" tb2 "https://github.com/harbor-framework/terminal-bench-2"
    for t in "${TB2_TASKS[@]}"; do
        local target="$OUT/tb2-$t.sif"
        if [[ -s "$target" && $FORCE -eq 0 ]]; then echo "SKIP tb2-$t (exists)"; continue; fi
        local extra=()
        [[ -n "${TB2_RUN[$t]:-}" ]] && extra=(--run "${TB2_RUN[$t]}")
        echo "==> building tb2-$t"
        python3 scripts/build_tb2_image.py "$t" \
            --upstream "$TB2_REPO" --apt "${TB2_APT[$t]:-}" "${extra[@]}" \
            --out "$target"
    done
}

# --- OpenThoughts-TBLite ---------------------------------------------------
#
# The paper's five TBLite tasks. Each replays its upstream Dockerfile; the
# generator handles the unprivileged apt shim and the verifier runtime, so no
# per-task recipe is needed here.
TBLITE_TASKS=(book-portfolio-analysis broken-python corrupted-filesystem-recovery
              cosign-keyless-signing reverse-engineer-stack-vm)

build_tblite() {
    require_clone "$TBLITE_REPO" "$TBLITE_COMMIT" tblite "https://github.com/open-thoughts/OpenThoughts-TBLite"
    local tasksdir="$TBLITE_REPO/tasks"
    [[ -d "$tasksdir" ]] || tasksdir="$TBLITE_REPO"
    for t in "${TBLITE_TASKS[@]}"; do
        local target="$OUT/tb-$t.sif"
        if [[ -s "$target" && $FORCE -eq 0 ]]; then echo "SKIP tb-$t (exists)"; continue; fi
        local taskdir="$tasksdir/$t"
        [[ -d "$taskdir" ]] || { echo "tblite task dir missing: $taskdir" >&2; exit 1; }
        echo "==> building tb-$t"
        python3 scripts/build_tblite_image.py "$taskdir" --out "$target"
    done
}

# --- drive -----------------------------------------------------------------
case "$FAMILY" in
    base)   build_agent_stateful; build_gitpush; build_db_entry ;;
    tb2)    build_tb2 ;;
    tblite) build_tblite ;;
    all)    build_agent_stateful; build_gitpush; build_db_entry; build_tb2; build_tblite ;;
esac

echo "=== done ==="
ls -la "$OUT"/*.sif 2>/dev/null | awk '{print $5, $9}'
