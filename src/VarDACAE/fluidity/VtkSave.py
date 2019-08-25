import numpy as np

from VarDACAE.fluidity import vtktools

class VtkSave():

    """Class to hold Fluidity saver helper functions.
    In theory this should be part of
    vtktools.py but I have kept it seperate to avoid confusion
    as to which is my work """

    def __init__(self):
        pass


    def save_structured_vtu(self, filename, struc_grid):
        from evtk.hl import pointsToVTK

        filename = filename.replace(".vtu", "")
        xs, ys, zs = self.__get_grid_locations(struc_grid)
        data = self.__get_grid_data(struc_grid)

        pointsToVTK(filename, xs, ys, zs, data)


    @staticmethod
    def save_vtu_file(arr, name, filename, sample_fp=None):
        """Saves an unstructured VTU file - NOTE TODO - should be using deep copy method in vtktools.py -> VtuDiff()"""
        if sample_fp == None:
            sample_fp = vda.get_sorted_fps_U(self.settings.DATA_FP)[0]

        ug = vtktools.vtu(sample_fp) #use sample fp to initialize positions on grid

        ug.AddScalarField('name', arr)
        ug.Write(filename)



    @staticmethod
    def __get_grid_locations(grid):
        npoints = grid.GetNumberOfPoints()
        xs = np.zeros(npoints)
        ys = np.zeros(npoints)
        zs = np.zeros(npoints)

        for i in range(npoints):
            loc = grid.GetPoint(i)
            xs[i], ys[i], zs[i] = loc
        return xs, ys, zs

    def __get_grid_data(self, grid):
        """Gets data and returns as dictionary (i.e. in form necessary for EVTK) """
        npoints = grid.GetNumberOfPoints()
        pointdata=grid.GetPointData()

        data = {}
        for name in self.__get_field_names(grid):
            vtkdata = pointdata.GetScalars(name)
            np_arr = nps.vtk_to_numpy(vtkdata)
            if len(np_arr.shape) == 1: #i.e. exclude vector fields
                data[name] = np_arr

        return data

    @staticmethod
    def __get_field_names(grid):
        vtkdata=grid.GetPointData()
        return [vtkdata.GetArrayName(i) for i in range(vtkdata.GetNumberOfArrays())]


