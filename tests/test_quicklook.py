from quicklook import quicklook
import numpy as np
import pytest


def test_static():
    
    # returns a numpy array of correct shape
    arr = quicklook.static(width=10, height=10)
    assert isinstance(arr, (np.ndarray))
    assert arr.shape == (10,10)

    # error raised if shape parameter input is not integer
    with pytest.raises(TypeError) as e_info:
      width = 'not an integer'
      height = 3.14159
      _ = quicklook.static(width=width, height=height)

def test_show():

  # error raised if array input is not a numpy array
  with pytest.raises(TypeError) as e_info:
    
    a = 'not_an_array'
    b = 1

    quicklook.show(a)
    quicklook.show(b)
  
def test_reshape_array():

  def test_3D_shape(arr):
    """3D arrays -> 3D with channel last ordering"""
    arr = quicklook.reshape_array(arr)
    ndim, shape = (arr.ndim, arr.shape)
    assert ndim == 3
    assert np.min(shape) == shape[-1] 
  
  # 1D array -> not allowed (!)
  with pytest.raises(ValueError) as e_info:
    arr_1D = np.array([1,2,3])
    _ = quicklook.reshape_array(arr_1D) 

  # 2D array (1 channel) -> 2D array
  arr_2D = np.zeros((10,10))
  ndim = quicklook.reshape_array(arr_2D).ndim
  assert ndim == 2

  # 3D array (1 channel) -> 2D array
  arr_3D_flat = np.zeros((10,10, 1))
  ndim = quicklook.reshape_array(arr_3D_flat).ndim
  assert ndim == 2

  # 3D arrays (2 channel) -> 2D array
  assert quicklook.reshape_array(np.zeros((2,10,10))).ndim == 2
  assert quicklook.reshape_array(np.zeros((10,2,10))).ndim == 2
  assert quicklook.reshape_array(np.zeros((10,2,10))).ndim == 2
  
  # 3D array (3 channels) -> 3D with channel last ordering
  test_3D_shape(np.zeros((3,10,10)))  # band first
  test_3D_shape(np.zeros((10,3,10)))  # band middle
  test_3D_shape(np.zeros((10,10,3)))  # band last

  # 3D array (4 channels) -> 3D with channel last ordering
  test_3D_shape(np.zeros((4,10,10)))  # band first
  test_3D_shape(np.zeros((10,4,10)))  # band middle
  test_3D_shape(np.zeros((10,10,4)))  # band last

  # high dimensional array
  with pytest.raises(ValueError) as e_info:
    arr_nD = np.zeros((2,2,2,2))
    _ = quicklook.reshape_array(arr_nD) 
