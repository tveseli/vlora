"""vlora CLI — command-line interface for adapter management."""

from __future__ import annotations

from pathlib import Path

import click

from vlora.io import LoRAWeights, load_adapter, save_adapter
from vlora.ops import explained_variance_ratio
from vlora.subspace import SharedSubspace


@click.group()
@click.version_option(package_name="vlora")
def cli():
    """vLoRA — Shared low-rank subspaces for LoRA adapter management."""
    pass


@cli.command()
@click.argument("subspace_path", type=click.Path(exists=True))
def info(subspace_path: str):
    """Show subspace stats: tasks, layers, compression ratios."""
    sub = SharedSubspace.load(subspace_path)

    click.echo(f"\n  Subspace: {subspace_path}")
    click.echo(f"  Components (k): {sub.num_components}")
    click.echo(f"  LoRA rank: {sub.rank}")
    click.echo(f"  Layers: {len(sub.layer_names)}")
    click.echo(f"  Tasks: {len(sub.tasks)}")

    if sub.tasks:
        click.echo(f"\n  Task IDs:")
        for tid in sorted(sub.tasks.keys()):
            click.echo(f"    - {tid}")

    # Variance explained (average across layers)
    first_layer = sub.layer_names[0]
    var_a = explained_variance_ratio(sub.singular_values_a[first_layer])
    var_b = explained_variance_ratio(sub.singular_values_b[first_layer])

    k = sub.num_components
    click.echo(f"\n  Variance explained (first layer, k={k}):")
    if k <= len(var_a):
        click.echo(f"    A: {var_a[k-1]:.1%}")
    if k <= len(var_b):
        click.echo(f"    B: {var_b[k-1]:.1%}")

    # Compression estimate
    n = len(sub.tasks)
    if n > 0 and sub.layer_names:
        dim_a = sub.components_a[first_layer].shape[1]
        dim_b = sub.components_b[first_layer].shape[1]
        n_layers = len(sub.layer_names)

        full_bytes = n * n_layers * (dim_a + dim_b) * 4  # float32
        basis_bytes = n_layers * k * (dim_a + dim_b) * 4
        loadings_bytes = n * n_layers * k * 2 * 4  # A + B loadings
        vlora_bytes = basis_bytes + loadings_bytes

        full_mb = full_bytes / 1e6
        vlora_mb = vlora_bytes / 1e6
        ratio = full_bytes / vlora_bytes if vlora_bytes > 0 else 0

        click.echo(f"\n  Storage ({n} adapters):")
        click.echo(f"    Full:  {full_mb:,.1f} MB")
        click.echo(f"    vLoRA: {vlora_mb:,.1f} MB")
        click.echo(f"    Ratio: {ratio:.1f}x")

    click.echo()


@cli.command()
@click.argument("adapter_dirs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory for shared subspace.")
@click.option("-k", "--num-components", type=int, default=None, help="Number of basis components (auto if not set).")
@click.option("--variance-threshold", type=float, default=0.6, help="Variance threshold for auto k selection.")
def compress(adapter_dirs: tuple[str, ...], output: str, num_components: int | None, variance_threshold: float):
    """Build shared subspace from adapter directories."""
    click.echo(f"\n  Loading {len(adapter_dirs)} adapters...")

    adapters = []
    task_ids = []
    for d in adapter_dirs:
        path = Path(d)
        adapters.append(load_adapter(path))
        task_ids.append(path.name)
        click.echo(f"    Loaded: {path.name}")

    click.echo(f"  Building subspace...")
    sub = SharedSubspace.from_adapters(
        adapters,
        task_ids=task_ids,
        num_components=num_components,
        variance_threshold=variance_threshold,
    )

    sub.save(output)
    click.echo(f"  Saved to: {output}")
    click.echo(f"  Components: {sub.num_components}, Layers: {len(sub.layer_names)}, Tasks: {len(sub.tasks)}")
    click.echo()


@cli.command("export")
@click.argument("subspace_path", type=click.Path(exists=True))
@click.argument("task_id")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory for PEFT adapter.")
def export_cmd(subspace_path: str, task_id: str, output: str):
    """Reconstruct a task adapter to PEFT format."""
    sub = SharedSubspace.load(subspace_path)

    if task_id not in sub.tasks:
        available = ", ".join(sorted(sub.tasks.keys()))
        raise click.ClickException(f"Unknown task '{task_id}'. Available: {available}")

    click.echo(f"\n  Reconstructing '{task_id}'...")
    weights = sub.reconstruct(task_id)
    save_adapter(weights, output)
    click.echo(f"  Exported to: {output}")
    click.echo()


@cli.command()
@click.argument("subspace_path", type=click.Path(exists=True))
@click.argument("adapter_dir", type=click.Path(exists=True))
@click.option("--task-id", required=True, help="ID for the new task.")
def add(subspace_path: str, adapter_dir: str, task_id: str):
    """Absorb a new adapter into an existing subspace."""
    sub = SharedSubspace.load(subspace_path)

    click.echo(f"\n  Loading adapter from {adapter_dir}...")
    adapter = load_adapter(adapter_dir)

    click.echo(f"  Absorbing as '{task_id}'...")
    sub.absorb(adapter, task_id)

    sub.save(subspace_path)
    click.echo(f"  Subspace updated. Tasks: {len(sub.tasks)}")
    click.echo()


@cli.command()
@click.argument("adapter_dirs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--threshold", type=float, default=0.9, help="Similarity threshold for clustering.")
def analyze(adapter_dirs: tuple[str, ...], threshold: float):
    """Analyze adapter similarity and find redundant clusters."""
    from vlora.analysis import compute_similarity_matrix, find_clusters

    click.echo(f"\n  Loading {len(adapter_dirs)} adapters...")

    adapters = []
    names = []
    for d in adapter_dirs:
        path = Path(d)
        adapters.append(load_adapter(path))
        names.append(path.name)
        click.echo(f"    Loaded: {path.name}")

    if len(adapters) < 2:
        raise click.ClickException("Need at least 2 adapters for analysis.")

    click.echo(f"\n  Computing similarity matrix...")
    sim = compute_similarity_matrix(adapters)

    # Print similarity matrix
    click.echo(f"\n  Pairwise Cosine Similarity:")
    header = "  " + " " * 20 + "  ".join(f"{n[:8]:>8}" for n in names)
    click.echo(header)
    for i, name in enumerate(names):
        row = f"  {name[:20]:<20}"
        for j in range(len(names)):
            val = sim[i, j].item()
            row += f"  {val:8.3f}"
        click.echo(row)

    # Clustering
    clusters = find_clusters(sim, threshold=threshold)
    click.echo(f"\n  Clusters (threshold={threshold}):")
    for ci, cluster in enumerate(clusters):
        members = ", ".join(names[i] for i in cluster)
        click.echo(f"    Cluster {ci + 1}: {members}")

    redundant = sum(len(c) - 1 for c in clusters if len(c) > 1)
    if redundant > 0:
        click.echo(f"\n  Potentially redundant adapters: {redundant}")
    else:
        click.echo(f"\n  No redundant adapters found at threshold={threshold}")

    click.echo()
