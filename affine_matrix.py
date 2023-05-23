import os
import numpy as np
import pydicom
import pylab
import itk
 
def get_affine_matrix(dicom_dir):
    '''
    Calculate the affine matrix which maps the voxel from the DICOM voxel space (column, row, depth) to patient
    coordinate system (x, y, z)
    reference: https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-slice-affine
    written by Shumao Pang, pangshumao@126.com
    :param dicom_dir: a dicom series dir
    :return: a numpy array with shape of (4, 4)
    '''
    
    dicom_file_list = os.listdir(dicom_dir)
    slice_num = len(dicom_file_list)
    dicom_list = [0] * slice_num
    # import pdb; pdb.set_trace()
    for dicom_file in dicom_file_list:
        ds = pydicom.read_file(os.path.join(dicom_dir, dicom_file))
        instance_number = ds.InstanceNumber
        dicom_list[instance_number - 1] = ds

    rows = dicom_list[0].Rows
    columns = dicom_list[0].Columns
    pixel_spacing = dicom_list[0].PixelSpacing
    image_orientation_patient = dicom_list[0].ImageOrientationPatient
    orientation_matrix = np.reshape(image_orientation_patient, [3, 2], order='F')
    # orientation_matrix = orientation_matrix[:, ::-1]
    print("orientation_matrix:\n", orientation_matrix)
 
    first_image_position_patient = np.array(dicom_list[0].ImagePositionPatient)
    last_image_position_patient = np.array(dicom_list[-1].ImagePositionPatient)
    k = (last_image_position_patient - first_image_position_patient) / (slice_num - 1)
 
    # affine_matrix = np.zeros((4, 4), dtype=np.float32)
    # affine_matrix[:3, 0] = orientation_matrix[:, 0] * pixel_spacing[0]
    # affine_matrix[:3, 1] = orientation_matrix[:, 1] * pixel_spacing[1]
    # affine_matrix[:3, 2] = k
    # affine_matrix[:3, 3] = first_image_position_patient

    ## ref: https://github.com/dgobbi/vtk-dicom/blob/master/Source/vtkDICOMReader.cxx
    if k[2] < 0:
        k[2] = -1
    else: 
        k[2] = 1
    affine_matrix = np.zeros((4, 4), dtype=np.float32)
    affine_matrix[:3, 0] = orientation_matrix[:, 0] 
    affine_matrix[:3, 1] = orientation_matrix[:, 1] * (-1) 
    affine_matrix[:3, 2] = k 
    affine_matrix[:3, 3] = first_image_position_patient + pixel_spacing[1] * (rows - 1) * orientation_matrix[:, 1]
    
    affine_matrix[3, 3] = 1.0
    return affine_matrix


if __name__ == '__main__':
    # data_dir = '/data/dicom_case1'
    # data_dir = "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val\\20170831-000005\\dicom"

    data_dir = "/home/th/Data/data_sphere_test/52/dicom"
    affine_matrix = get_affine_matrix(data_dir)
    inv_affine_matrix = np.linalg.inv(affine_matrix)
    import pdb; pdb.set_trace()
    print('affine matrix:\n', affine_matrix)
    print('inv affine matrix:\n', inv_affine_matrix)
 
    image_coord = np.array([0, 0, 0]) # (row, column, depth)
    input = np.ones(4, dtype=np.float32)
    input[:3] = image_coord
    # map the image coord to patient coord
    patient_coord = np.matmul(affine_matrix, input)
    print('image coord:\n', image_coord)
    print('patient coord:\n', patient_coord)
 
    # map the patient coord to image coord
    inv_image_coord = np.matmul(inv_affine_matrix, patient_coord).astype(int)[:3] # (row, column, depth)
    print('inv image coord:\n', inv_image_coord)
 
 