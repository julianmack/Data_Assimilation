
from VarDACAE.fluidity import VtkSave
from VarDACAE.fluidity import vtktools

def save_vtu(settings, fp, data, name=None):
    if settings.THREE_DIM:
        #structured vtu grid
        vtksaver = VtkSave()
        s_grid = vtktools.vtu(settings.VTU_FP).ugrid
        vtksaver.save_structured_vtu(fp, s_grid, data)
    else:
        #Save .vtu files so that I can look @ in paraview
        sample_fp = settings.get_loader().get_sorted_fps_U(self.settings.DATA_FP)[0]
        if not name:
            raise ValueError("Must provide name")
        VtkSave.save_vtu_file(data, name, fp, sample_fp)

