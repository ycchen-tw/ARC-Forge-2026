set -x
export HOME=$HOME
$HOME/.local/bin/uvx --from kaggle kaggle models create -p /dev/shm/uid-$UID/arc2r4at30-patch/upload/model
$HOME/.local/bin/uvx --from kaggle kaggle models instances create -p /dev/shm/uid-$UID/arc2r4at30-patch/upload/inst
echo EXIT=$?
