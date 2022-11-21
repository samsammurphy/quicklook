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
  

#TODO
# def test_percentile_clip():
#   add test here
