import functools
from xarray import DataArray
import pytest
from ..models import pytorch_custom, keras_vgg19, tfslim_custom, tfslim_vgg16


@pytest.mark.parametrize(['model_ctr', 'expected_num_weights'], [
    (pytorch_custom, 98569056),
    (functools.partial(pytorch_custom, kernel_size=5), 96801152),
    pytest.param(keras_vgg19, 143667240, marks=pytest.mark.memory_intense),
    (tfslim_custom, 1048296),
    pytest.param(tfslim_vgg16, 138361641, marks=pytest.mark.memory_intense),
])
def test_weights(model_ctr, expected_num_weights):
    model = model_ctr()
    weights = model.weights
    assert isinstance(weights, DataArray)
    assert len(weights) == expected_num_weights


@pytest.mark.parametrize(['model_ctr', 'expected_num_units'], [
    (functools.partial(pytorch_custom, kernel_size=5), 97800),
    pytest.param(keras_vgg19, 14861288, marks=pytest.mark.memory_intense),
    (tfslim_custom, 187624),
    pytest.param(tfslim_vgg16, 13556713, marks=pytest.mark.memory_intense),
])
def test_units(model_ctr, expected_num_units):
    model = model_ctr()
    units = model.units
    assert isinstance(units, DataArray)
    assert len(units) == expected_num_units
