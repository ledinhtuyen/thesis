from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg_custom.datasets.public_dataset import PublicDataset
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

def build_test_dataset(data_prefix, pipeline):
    return PublicDataset(
                  data_prefix=dict(
                      img_path=data_prefix['img_path'],
                      seg_map_path=data_prefix['seg_map_path']
                  ),
                  pipeline=pipeline
                )

if __name__ == "__main__":
  # Test pipeline
  test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
  ]
  
  # Build test dataset
  test_dataset = build_test_dataset(
                  data_prefix=dict(
                      img_path='/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ClinicDB/images',
                      seg_map_path='/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ClinicDB/masks'
                  ),
                  pipeline=test_pipeline
                )

  # Config and checkpoint path
  config = "configs/mae/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512.py"
  ckpt = "/home/s/tuyenld/mae/downstream/segmentation/work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp7/iter_24000.pth"

  # Init model
  model = init_model(config, ckpt, device='cuda:0')
  
  # save_dir
  save_dir = 'infer'
  
  # Inference on multiple images
  for i in range(int(len(test_dataset))):
    index = i
    # Inference on single image
    result = inference_model(model, test_dataset.get_data_info(index)["img_path"])
    
    # Name of the image
    name_image = test_dataset.get_data_info(index)["img_path"].split("/")[-1]
    
    # Add gt_sem_seg to result
    result.gt_sem_seg = test_dataset[index]["data_samples"].gt_sem_seg
    
    # Save image to infer folder
    vis_image = show_result_pyplot(
      model, 
      test_dataset.get_data_info(index)["img_path"], 
      result, 
      title="Polyp Segmentation",
      with_labels=False,
      show=False, 
      save_dir=save_dir, 
      out_file=f"{save_dir}/{name_image}")
    print(f"Save image to {save_dir}/{name_image}")
