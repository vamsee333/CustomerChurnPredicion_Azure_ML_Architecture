
import pytest
from notebooks.Pytests import test_invoke_service



@pytest.fixture

def pass_variable():
    x = 5
    return x


def test_pipeline(pass_variable):
    res = test_invoke_service.test_invoke_service(pass_variable)
    assert res == 25

def test_pipeline_fail(pass_variable):
    res = test_invoke_service.test_invoke_service(pass_variable)
    assert res == 4
