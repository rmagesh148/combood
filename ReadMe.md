- The 4 most important files for this paper are test_ood.py, utils.py, data.py, confidenciator.py   

- Download the OpenOOD (Git repo: https://github.com/Jingkang50/OpenOOD/tree/main)  datasets  and checkpoints from this link:
 https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eso7IDKUKQ9AoY7hm9IU2gIBMWNnWGCYPwClpH0TASRLmg?e=kMrkVQ 

-  Run the  test_ood.py file to  check results.  

- Make sure you provide necessary directory for each dataset and checkpoint(pretrained models) before running the code.  

-  Change the directory of  "OpenOOD " (/confidence-magesh_MR/confidence-magesh/OpenOOD) folder inside load.py (/confidence-magesh/models/load.py) script.   

- Similarly change the directory of "OpenOOD" folder inside data.py (/confidence-magesh/data.py) script.  

To run it with Cifar10:

- Provide necessary directories of pretrained OpenOOD checkpoint models inside the script: /confidence-magesh/OpenOOD/openood_id_ood_and_model_cifar10.py

- Provide necessary directories of  OpenOOD datasets  inside the files: 
/home/saiful/confidence_icdb/confidence-magesh/OpenOOD/configs/datasets/cifar10/cifar10.yml 
and  /home/saiful/confidence_icdb/confidence-magesh/OpenOOD/configs/datasets/cifar10/cifar10_ood.yml.

Please follow the similar approach to run it with mnist, cifar100, and imagenet. 
You need to provide directory of the OpenOOD  datasets and checkpoints inside: 
	/confidence-magesh/OpenOOD/openood_id_ood_and_model_mnist.py,  
	/confidence-magesh/OpenOOD/openood_id_ood_and_model_cifar100.py,  
	and confidence-magesh/OpenOOD/openood_id_ood_and_model_imagenet.py files. 

- The results can be found inside the following directories: 
	for mnist : /confidence-magesh/results/mnist_lenet/knn/
	for cifar10: /confidence-magesh/results/cifar10_resnet/knn/
	for cifar100: /confidence-magesh/results/cifar100_resnet/knn/
	for imagenet: /confidence-magesh/results/imagenet_resnet50/knn/
	for document: /confidence-magesh/results/document_resnet50_docu/knn/	


To run it with Document dataset:

- Download the dataset from this link https://adamharley.com/rvl-cdip/ 
- Preprocess the dataset  folder directories following this link https://github.com/MdSaifulIslamSajol/mobilenet_image_classification_with_document_dataset/blob/main/make_classwise_subfolders_rvl_cdip.py 
- The processed dataset can also directly  be downloaded from this link https://lsu.box.com/s/x71r0eiagqgbqxei50ghbk9cldslxr34
- The pretrained checkpoints of Resnet50 for document dataset can be found on this directory: /confidence-magesh/document classification/saved trained models/resnet50_checkpoints/resnet50_acc0.9_epoch40_on_319837_trainimages_load.ckpt"
- The OOD datasets for document dataset can be found on this link: https://github.com/gxlarson/rvl-cdip-ood 
- Now provide directory of the OpenOOD  datasets and checkpoints inside: confidence-magesh/document_id_ood_n_model_loader.py script .


bibliography: bibliography.bib

Citation:
---

---
