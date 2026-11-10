"""Package a private recovery overlay without changing the frozen data archive."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from genie_diep.baseline import implementation_hash, validate_readiness
from genie_diep.data import file_hash
from scripts.build_recovery_verification_notebook import REQUIRED_MEMBERS, build
from scripts.gpu_recovery_verification import model_source_hash


def run(output, checkpoint, original_spec):
    root=Path.cwd();output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous verification exports')
    validate_readiness(root,'configs/training-baseline-v1.json')
    sources={name:root/name for name in REQUIRED_MEMBERS}
    sources['recovery-inputs/last.pt']=Path(checkpoint)
    sources['recovery-inputs/original-spec.json']=Path(original_spec)
    for path in sources.values():
        if not path.is_file():raise FileNotFoundError(path)
    output.mkdir(parents=True)
    archive=output/'genie-recovery-verification-v1.zip';hashes={}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as bundle:
        for name,path in sorted(sources.items()):
            hashes[name]=file_hash(path);bundle.write(path,name)
        bundle.writestr('overlay-sha256.json',json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist())==set(hashes)|{'overlay-sha256.json'}
        for name,digest in hashes.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest()==digest,name
    manifest=dict(overlay_name=archive.name,overlay_sha256=file_hash(archive),overlay_bytes=archive.stat().st_size,
        expected_checkpoint_sha256=hashes['recovery-inputs/last.pt'],
        expected_implementation_sha256=implementation_hash(),
        expected_spec_sha256=hashes['configs/training-baseline-v1.json'],expected_model_sha256=model_source_hash(root),
        member_hashes=hashes,contains_private_diagnostic_weights=True,contains_dataset=False)
    path=output/'overlay-manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    build(output/'launcher',path)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--original-spec',required=True)
    args=parser.parse_args();run(args.out,args.checkpoint,args.original_spec)
