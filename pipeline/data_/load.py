import numpy as np
import random

import os

from pipeline.fluidity import VtkSave, vtktools
from pipeline.data_.augmentation import FlipHorizontal, FieldJitter
from pipeline.data_.split import SplitData

from vtk.util import numpy_support as nps


from torch.utils.data import TensorDataset, DataLoader

from torchvision import transforms
import torch

class Data3D_Dataset(TensorDataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
    def __getitem__(self, index):
        sample = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            sample = self.transform(sample)
        return sample

class GetData():
    """Class to load data from files in preparation for Data Assimilation or AE training"""
    def __init__(self):
        pass

    def get_train_test_loaders(self, settings, batch_sz, num_workers = 6, small_debug=False):


        X = self.get_X(settings)

        splitter = SplitData()
        train_X, test_X, DA_u_c, X_norm,  mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

        if small_debug: #take v. small subset of test and train (for speed)
            batch_sz = 3 #to test sizes
            num_workers = 0
            train_X = train_X[-8:]
            test_X = test_X[:8]

        #Add Channel if we are in 3D case
        if settings.THREE_DIM:
            train_X = np.expand_dims(train_X, 1)
            test_X = np.expand_dims(test_X, 1)

        #ADD Data Augmentation to train_loader
        trnsfrm = None
        if hasattr(settings, "AUGMENTATION") and settings.AUGMENTATION:
            trnsfrm = transforms.Compose([
                            transforms.RandomApply([FlipHorizontal("x"),
                                                    FlipHorizontal("y")], p=0.5),
                            transforms.RandomChoice([FieldJitter(0.01, 0.1),
                                                    FieldJitter(0.005, 0.5),
                                                    FieldJitter(0., 0.)], ),
                            ])


        #Dataloaders
        train_dataset = Data3D_Dataset(torch.Tensor(train_X), transform=trnsfrm)
        train_loader = DataLoader(train_dataset, batch_sz, shuffle=True, num_workers=num_workers)
        test_dataset = Data3D_Dataset(torch.Tensor(test_X)) #no augmentation for test set
        test_batch_sz = min(test_X.shape[0], batch_sz)
        test_loader = DataLoader(test_dataset, test_batch_sz)

        # for batch_idx, data in enumerate(train_loader):
        #     x, = data
        #     print(batch_idx, x.shape)
        # exit()
        
        #save train_X and test_X
        self.train_X = train_X
        self.test_X = test_X

        return train_loader, test_loader

    def get_X(self, settings):
        """Returns X in the M x n format"""
        if not os.path.exists(settings.get_X_fp()) or settings.FORCE_GEN_X:
            X = None
            if settings.AZURE_DOWNLOAD:
                try:
                    X = GetData.download_X_azure(settings)
                except:
                    pass
            if not X:
                fps = self.get_sorted_fps_U(settings.DATA_FP)
                X = self.create_X_from_fps(fps, settings)
        else:
            X = np.load(settings.get_X_fp(),  allow_pickle=True)

        return X

    @staticmethod
    def get_sorted_fps_U(data_dir, max = 988):
        """Creates and returns list of .vtu filepaths sorted according
        to timestamp in name.
        Input files in data_dir must be of the
        form <XXXX>LSBU_<TIMESTEP INDEX>.vtu"""

        fps = os.listdir(data_dir)

        #extract index of timestep from file name
        idx_fps = []
        for fp in fps:
            _, file_number = fp.split("LSBU_")
            #file_number is of form '<IDX>.vtu'
            idx = int(file_number.replace(".vtu", ""))
            idx_fps.append(idx)


        #sort by timestep
        assert len(idx_fps) == len(fps)
        zipped_pairs = zip(idx_fps, fps)
        fps_sorted = [x for _, x in sorted(zipped_pairs)]

        #add absolute path
        fps_sorted = [data_dir + x for x in fps_sorted]

        return fps_sorted

    @staticmethod
    def create_X_from_fps(fps, settings, field_type  = "scalar"):
        """Creates a numpy array of values of scalar field_name
        Input list must be sorted"""

        M = len(fps) #number timesteps

        for idx, fp in enumerate(fps):
            # create array of tracer
            ug = vtktools.vtu(fp)
            if not settings.THREE_DIM:
                matrix = GetData.get_1D_np_from_ug(ug,  settings.FIELD_NAME, field_type)
            elif settings.THREE_DIM == True:
                matrix = GetData.get_3D_np_from_ug(ug, settings)
            else:
                raise ValueError("<config>.THREE_DIM must be True or eval to False")
            mat_size = matrix.shape

            if idx == 0:
                #fix length of vectors and initialize the output array:
                n = matrix.shape
                size = (M,) + n
                output = np.zeros(size)
            else:
                #enforce all vectors are of the same length
                assert mat_size == n, "All input .vtu files must be of the same size."
            output[idx] = matrix

        #return (M x nx x ny x nz) or (M x n)
        if settings.SAVE:
            np.save(settings.get_X_fp(), output, allow_pickle=True)

        return output


    def download_X_azure(settings):
        fp_azure = settings.get_X_fp().replace(settings.INTERMEDIATE_FP, "")
        try:
            os.makedirs(settings.INTERMEDIATE_FP)
        except FileExistsError:
            pass
        GetData.__donwload_azure_blob(settings, settings.get_X_fp(), fp_azure)
        X = np.load(settings.get_X_fp(), allow_pickle=True)
        return X

    def __donwload_azure_blob(settings, fp_save, fp_to_access):
        from azure.storage.blob import BlockBlobService
        #i.e. import here ^^ as this will only be used once (at download time)

        block_blob_service = BlockBlobService(account_name=settings.AZURE_STORAGE_ACCOUNT,
                                        account_key=settings.AZURE_STORAGE_KEY)
        block_blob_service.get_blob_to_path(settings.AZURE_CONTAINER, fp_to_access,
                                            fp_save)

    @staticmethod
    def get_1D_np_from_ug(ug, field_name, field_type = "scalar"):
        if field_type == "scalar":
            matrix = ug.GetScalarField(field_name)
        elif field_type == "vector":
            matrix = ug.GetVectorField(field_name)
        else:
            raise ValueError("field_name must be in {\'scalar\', \'vector\'}")
        return matrix

    @staticmethod
    def get_3D_np_from_ug(ug, settings, save_newgrid_fp=None):
        """Returns numpy array or torch tensor of the vtu file input
        Accepts:
            :ug - an unstructured grid .vtu object
            :setting - a Config object containing some/all of the following information:
                :FACTOR_INCREASE - Factor by which to increase (or decrease) the number of points (when newshape=None)
                     the number of output points is (approximately) the (input number points * FACTOR_INCREASE)
                :n - tuple of 3 ints which gives new shape of output. Overides FACTOR_INCREASE
            :save_newgrid_fp - str. if not None, the restructured vtu grid will be
                saved at this location relative to the working directory"""

        field_name = settings.FIELD_NAME

        newshape = GetData.__get_newshape_3D(ug, settings.get_n(), settings.FACTOR_INCREASE, )

        #update settings
        settings.n3d = newshape

        (nx, ny, nz) = newshape

        # Get structured grid from unstructured grid using newshape
        # This will interpolate between points in the unstructured grid
        struct_grid = ug.StructuredPointProbe(nx, ny, nz)

        if save_newgrid_fp: # useful for debug
            VtkSave.save_structured_vtu(save_newgrid_fp, struct_grid)

        pointdata = struct_grid.GetPointData()
        vtkdata = pointdata.GetScalars(field_name)
        np_data = nps.vtk_to_numpy(vtkdata)

        #Fortran order reshape (i.e first index changes fastest):
        result = np.reshape(np_data, newshape, order='F')


        return result

    @staticmethod
    def __get_newshape_3D(ug, newshape, factor_inc, ):

        if newshape == None:
            points = ug.ugrid.GetPoints()

            npoints = factor_inc * points.GetNumberOfPoints()

            bounds = points.GetBounds()
            ax, bx, ay, by, az, bz = bounds

            #ranges
            x_ran, y_ran, z_ran = bx - ax, by - ay, bz - az

            spacing = (x_ran * y_ran * z_ran / npoints)**(1.0 / 3.0)

            nx = int(round(x_ran / spacing))
            ny = int(round(y_ran / spacing))
            nz = int(round(z_ran / spacing))

            newshape = (nx, ny, nz)
        else:
            pass

        return newshape
