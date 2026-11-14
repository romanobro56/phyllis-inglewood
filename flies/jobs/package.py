"""Create a source-only GPU handoff; no checkpoints, histories or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
FLY=ROOT/'flies'/'jobs'


def source_files():
    allowed={'README.md','AUDIT.md','.gitignore','requirements.txt','profile.json','verified_profile.json',
             'elton_fly_job_links.txt','Elton_Inglewood_Fly_Resume.pdf','station.js','station_view.html'}
    files=[p for p in FLY.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in allowed)]
    files += [p for p in (FLY/'vendor').glob('*') if p.name in ('three.module.js','THREE-LICENSE.txt')]
    files += [p for p in (FLY/'readiness').glob('*.json') if p.name!='packaging_validation.json']
    if (FLY/'artifacts'/'README.md').is_file():files.append(FLY/'artifacts'/'README.md')
    report=ROOT/'GPU_TRAINING_READINESS.md'
    if report.is_file():files.append(report)
    return sorted(p for p in files if not p.is_symlink())


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,default=FLY/'artifacts'/'elton_workday_gpu_source.zip')
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(args.out,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            relative=path.relative_to(ROOT).as_posix()
            data=path.read_bytes()
            archive.writestr(relative,data)
            manifest.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        archive.writestr('TRANSFER_MANIFEST.json',json.dumps({'purpose':'Apply to a compatible Phyllis Inglewood repository; shared core/data/dependencies are not bundled','files':manifest},indent=2)+'\n')
    print(json.dumps({'archive':str(args.out),'source_files':len(manifest),'bytes':args.out.stat().st_size,'excludes':['model weights','training outputs','account/application state','credential vaults','Python caches']},indent=2))


if __name__=='__main__':main()
