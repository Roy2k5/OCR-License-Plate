PYTHON:=C:\Miniconda3\envs\ultralytics\python.exe
TRAIN_SCRIPT:=ocr_train.py
TEST_SCRIPT:=ocr_test.py
TASK:=non_ctc

# Config
BATCH_SIZE:=16
EPOCH:=100
LR:=1e-4
IS_CONTINUE:=--continue-train
.PHONY: train test

train:
	$(PYTHON) $(TRAIN_SCRIPT) --batch-size $(BATCH_SIZE) --lr $(LR) --epoch $(EPOCH) $(IS_CONTINUE)


test:
	$(PYTHON) $(TEST_SCRIPT) 
	type result_$(TASK).txt
	@$(PYTHON) -c "import cv2;import matplotlib.pyplot as plt;img = cv2.imread(\"accuracy_$(TASK).png\");img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB);plt.imshow(img);plt.axis(\"off\");plt.show()"
