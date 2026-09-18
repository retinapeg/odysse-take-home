"""Execute the canonical notebook (or a path given as the first argument) in a fresh kernel; preserve failures separately."""
from pathlib import Path
from datetime import datetime, timezone
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / 'notebooks' / 'Odysse_Ride_Analysis.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3', transport='ipc')
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, km=manager, timeout=600, resources={'metadata': {'path': str(root)}})
try:
    client.execute()
except Exception:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    failure = root / 'outputs' / f'failed_run_{stamp}.ipynb'
    failure.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, failure)
    raise
else:
    temp = path.with_suffix('.tmp.ipynb')
    nbformat.write(notebook, temp)
    temp.replace(path)
    executed = sum(c.cell_type == 'code' and c.execution_count is not None for c in notebook.cells)
    print(f'Executed {executed} code cells successfully: {path}')
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
