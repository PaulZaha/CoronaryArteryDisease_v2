from PIL import Image, ImageDraw
import os
import json



def mask_png(split,name):
    #Navigate to annotation json
    os.chdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'annotations'))

    #Read json data
    with open(split + ".json","r") as f:
        json_data = json.load(f)
    os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..'))
    segmentation_values = []
    id_list = []
    for images in json_data['images']:
        if images['file_name'] == name:
            id_list.append(images['id'])
            print(id_list)
    #Iterate through annotations, looking for image_id:name and save values in segmentation_values list
        for annotation in json_data['annotations']:
            if annotation['image_id'] in id_list:
                segmentation_values.append(annotation['segmentation'][0])


    img_size = (512,512)
    mask = Image.new('RGBA',img_size,(0,0,0,0))
    draw = ImageDraw.Draw(mask)

    for seg_val in segmentation_values:
        draw.polygon(seg_val,fill=(255,255,255,255))
    mask_save_path = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis', split, 'masks')

    
    print(name)
    print(os.getcwd())

    mask.save(os.path.join(mask_save_path,name[:-4] + "_mask.png"))

    
    

def main():

    splits = ['test', 'train','val']

    for split in splits:
        image_dir = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis', split, 'images')
        for image_name in os.listdir(image_dir):
            mask_png(split, image_name)


if __name__ == "__main__":
    main()
