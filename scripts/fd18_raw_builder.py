"""Original raw builder plus explicit decode availability metadata; pixels unchanged."""
# Role: runs scripts/21_build_fused_cache.py unchanged, with network connections disabled, and
#   also writes a per-clip availability record next to each test cache it builds.
# Used by: scripts/el25r_p1_repeat_runtime.py (step 3, once per view: det, miw, det248);
#   inference.
import sys,json,socket
from pathlib import Path
from importlib import import_module
# Package root; src/ and scripts/ go first on the import path.
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def main():
    from cuhkx.paths import p
    # The cache builder script as a module, and its build function.
    original=import_module('21_build_fused_cache');builder=original.build_fused_cache
    # Replacement for build_fused_cache: build as usual, then write the availability record.
    def wrapped(*args,**kwargs):
        # Build the cache; take the name suffix from the '--suffix=...' argument; per clip,
        # D = at least one frame stored, I = frames stored and IR readable (not in 'ir_dead').
        cache,report=builder(*args,**kwargs);suffix=next((v.split('=',1)[1] for v in sys.argv if v.startswith('--suffix=')),'');dead=set(report['ir_dead']);availability={str(c):dict(D=bool(cache.offsets[i+1]>cache.offsets[i]),I=bool(cache.offsets[i+1]>cache.offsets[i] and c not in dead)) for i,c in enumerate(cache.clip_ids)}
        # Write cache_root/fused<suffix>_test-availability.json with the flags and the build
        # report, then return the cache unchanged. Nothing in the package reads this file back;
        # the entry point recomputes availability from the raw folders.
        target=p('cache_root')/f'fused{suffix}_test-availability.json';target.write_text(json.dumps(dict(source='original build_fused_cache decode report',availability=availability,report=report)),encoding='utf-8');return cache,report
    # Replacement for socket connections: always raises.
    def denied(*args,**kwargs):raise RuntimeError('Raw builder network is disabled')
    # Disable network connections, install the replacement (the builder's main() looks the
    # function up by name when it calls it) and run that main() on this process's arguments.
    socket.socket.connect=denied;socket.create_connection=denied;original.build_fused_cache=wrapped;original.main()
if __name__=='__main__':main()
