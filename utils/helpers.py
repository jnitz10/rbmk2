import io
import sys


def capture_print(method, *args, **kwargs):
    # Redirect stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the method that prints
    method(*args, **kwargs)

    # Capture the output and restore stdout
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Return the captured output
    return output
