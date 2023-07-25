import os
import SimpleITK as sitk
import numpy as np
import shutil


if __name__ == "__main__":
    base_path = r"\\192.168.1.40\data\data03\dst"
    # dst_base_path = "/mnt/Data/data03_clean/dst/"
    # dst_base_path = "/home/th/Data/dulicui/project/data/data03_clean/dst"
    dst_base_path = r"\\192.168.1.40\data\data03_clean\dst"

    mask_base_path = os.path.join(base_path, "mask", "Sphere")
    dcm_base_path = os.path.join(base_path, "dicom")

    mask_dst_base_path = os.path.join(dst_base_path, "mask", "Sphere")
    dcm_dst_base_path = os.path.join(dst_base_path, "dicom")

    uids = os.listdir(mask_base_path)
    print(f"========uid: {len(uids)}")

    # import pdb; pdb.set_trace()
    for uid in uids[247:]:
        # import pdb; pdb.set_trace()
        print(f"dealing with: {uid}")
        uid_mask_path = os.path.join(mask_base_path, uid)

        cnt = 0
        for dcm in os.listdir(uid_mask_path):
            dcm_path = os.path.join(uid_mask_path, dcm)

            tmp = sitk.ReadImage(dcm_path)
            tmp = sitk.GetArrayFromImage(tmp)

            if np.max(tmp) != 0: ## have fg: sphere
                # import pdb; pdb.set_trace()
                mask_s = dcm_path

                mask_d_base = os.path.join(mask_dst_base_path, uid)
                mask_d = os.path.join(mask_d_base, dcm)

                if not os.path.exists(mask_d_base):
                    print(mask_d_base)
                    os.makedirs(mask_d_base)


                # os.system(f"ln -s {mask_s} {mask_d}")  ## softlink
                # os.system(f"cp {mask_s} {mask_d}")  ## softlink
                # os.system(f"mklink {mask_d} {mask_s}") ## window softlink
                


                image_s = os.path.join(dcm_base_path, uid, dcm)

                image_d_base = os.path.join(dcm_dst_base_path, uid)
                image_d = os.path.join(image_d_base, dcm)

                if not os.path.exists(image_d_base):
                    os.makedirs(image_d_base)

                # os.system(f"ln -s {image_s} {image_d}")  ## softlink
                # os.system(f"cp {image_s} {image_d}")
                # os.system(f"mklink {image_d} {image_s}")


                print(f"=========={dcm}--{cnt}")
                print(mask_s)
                print(mask_d)
                print(image_s)
                print(image_d)

                shutil.copy(mask_s, mask_d)
                shutil.copy(image_s, image_d)

                
                cnt += 1
            else:
                continue  ## give up mask/image with no anno 

