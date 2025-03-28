from argparse import ArgumentTypeError
import atheris
import numbers
import sys

MAX_DATA_LENGTH = 1000  # Maximum length of input data

default_path = "../../benchmarks"
sys.path.append(default_path)

with atheris.instrument_imports():
    import common.utils.validators


def TestOneInput(data):
    """The entry point for the fuzzer."""
    fdp = atheris.FuzzedDataProvider(data)

    # Test different input types
    str_value = fdp.ConsumeUnicode(count=MAX_DATA_LENGTH)
    int_value = fdp.ConsumeInt(MAX_DATA_LENGTH)
    float_value = fdp.ConsumeFloat()

    test_functions = (
        common.utils.validators.check_for_link,
        common.utils.validators.check_no_spaces,
        common.utils.validators.check_valid_filename,
        common.utils.validators.check_valid_folder,
        common.utils.validators.check_valid_file_or_dir,
        common.utils.validators.check_volume_mount,
        common.utils.validators.check_shm_size,
        common.utils.validators.check_num_cores_per_instance,
        common.utils.validators.check_positive_number,
        common.utils.validators.check_positive_number_or_equal_to_negative_one
    )

    for test_function in test_functions:
        for value in (str_value, int_value, float_value):
            try:
                test_function(value=value)
            except (ArgumentTypeError, ValueError, TypeError):  # Expected exception types
                pass
            except OverflowError as e:
                if isinstance(value, numbers.Number):
                    pass  # https://bugs.python.org/issue37436 (not a bug)
                else:
                    raise e


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
