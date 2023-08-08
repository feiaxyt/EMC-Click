python -m torch.distributed.launch --master_port=4321 --nproc_per_node=8 scripts/evaluate_model.py EMC-Click \
    --model_dir='./weights/' \
    --checkpoint=hr18s.pth,hr18.pth,hr32.pth,segb0.pth,segb3.pth \
    --n-clicks=20 \
    --gpus=0,1,2,3,4,5,6,7 \
    --target-iou=0.9 \
    --thresh=0.5 \
    --eval-mode='emc-click' \
    --datasets=GrabCut,Berkeley,SBD,DAVIS,PascalVOC
