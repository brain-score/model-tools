import functools
from xarray import DataArray
import pytest
from ..models import pytorch_custom, keras_vgg19


@pytest.mark.parametrize(['model_ctr', 'expected_num_weights'], [
    (pytorch_custom, 98569056),
    (functools.partial(pytorch_custom, kernel_size=5), 96801152),
    (keras_vgg19, 143667240),
])
def test_weights(model_ctr, expected_num_weights):
    model = model_ctr()
    weights = model.weights
    assert isinstance(weights, DataArray)
    assert len(weights) == expected_num_weights


@pytest.mark.parametrize(['model_ctr', 'expected_num_units'], [
    (functools.partial(pytorch_custom, kernel_size=5), 97800),
    (keras_vgg19, 14861288),
])
def test_units(model_ctr, expected_num_units):
    model = model_ctr()
    units = model.units
    assert isinstance(units, DataArray)
    assert len(units) == expected_num_units
