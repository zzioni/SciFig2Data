# finetune p5 models

cd yolov7
python train.py \
--workers 8 \
--device 0 \
--batch-size 32 \
--data data/custom.yaml \ #change
--img 640 640 \
--cfg cfg/training/yolov7-custom.yaml \ #change
--weights 'yolov7_training.pt' \
--name yolov7_subfigure_detection \ #change
--hyp data/hyp.scratch.custom.yaml \
--epoch 100

