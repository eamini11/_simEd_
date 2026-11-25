################################################################################
from typing import Any, Callable  # use Callable as a type hint

# global counters
class Tests:
    total = 0
    incorrect = 0

def printOneTest(_function: Callable, *args: tuple, **kwargs: dict) -> None:
    ''' Function to streamline fruitful-function testing, allowing the user to
        pass in a function, arbitrary number of arguments, and an expected result.

    Parameters:
        - _function: the name of a Callable function (the function name -- not a call)
        - (as part of *args): pass zero or more arguments, each separated by a comma,
            that your function will need when called
        - "expected" (as part of **kwargs): the expected result of the function call
        - [optional (as part of **kwargs)]: 
            - "context":   [default ""] additional context to add to printout
            - "fruitful":  [default True]  if False, student function is expected to print, not return
            - "simple":    [default False] if True, uses "+" and "x" for correct and incorrect
            - "indent":    [default False] if True, prints a tabstop print to each output line

    Example Usage:
        from printOneTest import *

        def blarb() -> int:
            return 2 + 4 + 6 + 8
        def sumOfInts(a: int, b: int, c: int) -> int:
            return a + b + c
        def printSum(a: int, b: int, c: int) -> None:
            print(f"{a} + {b} + {c} = {a + b + c}")

        def main() -> None:
            printOneTest(blarb, expected = 20)
            printOneTest(sumOfInts, 1, 2, 3, expected = 6)
            printOneTest(sumOfInts, 1, 2, 3, expected = 6, simple = True, indent = True)
            printOneTest(printSum, 1, 2, 3, expected = "1 + 2 + 3 = 6", fruitful = False)

        main()
    '''
    # check for 'indent' keyword argument, defaulting to False if not provided
    try:    _indent = kwargs["indent"]
    except: _indent = False
    _tab = '' if not _indent else '\t'

    try:    _context = " " + kwargs["context"]
    except: _context = ""

    # enforce that the first argument is a function reference, 
    # not the result of a function call
    if not isinstance(_function, Callable):
        print(f"{_tab}ERROR: first argument to printOneTest must be a callable, " + \
              f"not {type(_function).__name__}")
        return

    # grab the frame object associated with the student's calling module, and then import
    # that module (outside of this module) so we can call the function to be tested
    import inspect
    _calling_frame = inspect.stack()[-1] #[0]
    _student_module = inspect.getmodule(_calling_frame[0])
    _line_num = _calling_frame.lineno

    # determine whether the student is passing in a method or function
    # (method will have a .__self__; a function will raise AttributeError)
    try: 
        _function.__self__
    except AttributeError as err: 
        _is_method = False
    else:
        _is_method = True

    # build the function call string for later evaluating
    arg_string = ','.join(str(repr(_arg)) for _arg in args)
    _function_call_string = f"{_function.__name__}({arg_string})"

    # enforce a provided 'expected' keyword argument
    try: _expected = kwargs["expected"]
    except:
        print(f"{_tab}ERROR: must provide 'expected' keyword argument to printOneTest")
        return

    # check for 'fruitful' keyword argument, defaulting to True if not provided
    try:    _is_fruitful = bool(kwargs["fruitful"])
    except: _is_fruitful = True

    local_namespace: dict = locals()
    local_namespace[f"{_function.__name__}"] = _function

    try:
        if not _is_fruitful:
            import io
            from contextlib import redirect_stdout
            # student's function uses print rather than return
            _student_stdio =  io.StringIO()
            with redirect_stdout(_student_stdio):
                if _is_method:
                    eval(f"_function.__self__.{_function_call_string}", local_namespace)
                else:
                    eval(f"_student_module.{_function_call_string}", local_namespace)
            _result = _student_stdio.getvalue().strip() # result of student's print sans '\n'
        else:
            # fruitful function returns value
            if _is_method:
                _result = eval(f"_function.__self__.{_function_call_string}", local_namespace)
            else:
                _result = eval(f"_student_module.{_function_call_string}", local_namespace)
    except Exception as err:
        print(f"{_tab}ERROR in evaluating {_function_call_string}: {err}")
        import traceback
        traceback.print_exc()
        return

    _def = "\033[0m" # default
    _bg = {"red":"\033[37;41m", "grn":"\033[37;42m"}  # white text on background
    _check = f"{_bg['grn']}[✓]{_def}"; _ecks = f"{_bg['red']}[✗]{_def}"

    try:    _simple = kwargs["simple"]
    except: pass
    else: (_check, _ecks) = ("+", "x") if _simple else ("✓", "✗")

    # define a new __eq__ for lists
    def my__eq__(self, other) -> bool:
        if not isinstance(other, list): return False
        if len(self) != len(other):     return False
        for i in range(len(self)):
            if isinstance(self[i], float) or isinstance(other[i], float):
                if abs(float(self[i]) - float(other[i])) > 1e-6:
                    return False
            else:
                if self[i] != other[i]: return False
        return True

    if isinstance(_expected, float):
        # for floats, don't use ==; rather, check for "close enough"
        _is_correct = _check if abs(_result - _expected) < 1e-6 else _ecks
    elif isinstance(_expected, str):
        # if the student is using a non-fruitful function to print a floating
        # point result, then == may not work; so try to convert result to a
        # float and if successful, use "close enough" criteria; o/w treat as
        # arbitrary str
        try:     float(_result); float(_expected)
        except:  _is_correct = _check if _result == _expected else _ecks
        else:    _is_correct = _check if abs(float(_result) - float(_expected)) < 1e-6 else _ecks
    elif isinstance(_expected, list):
        _is_correct = _check if my__eq__(_result, _expected) else _ecks
    elif isinstance(_expected, dict) and isinstance(_expected[list(_expected.keys())[0]], list):
        # check for dict with list-values
        all_correct = _expected.keys() == _result.keys()
        if all_correct:
            for key in _expected.keys():
                all_correct &= my__eq__(_result[key], _expected[key])
        _is_correct = _check if all_correct else _ecks
    elif _expected is Any:
        _is_correct = _check
    else:
        _is_correct = _check if _result == _expected else _ecks

    max_arg_len = 70
    if len(arg_string) > max_arg_len: 
        comma_loc = arg_string.rfind(",", 0, max_arg_len); end = " ..."
        if comma_loc < 0: comma_loc = max_arg_len - 1; end = "..."
        short_arg_string = arg_string[:comma_loc + 1] + end
        _function_call_string = f"{_function.__name__}({short_arg_string})"

    if _is_method:
        # Consider:  <__main__.main.<locals>.ExampleClass object at 0x1003a7fa0
        # scrap "<__main__.main.<locals>." and just show object class type
        print(f"{_tab}Testing <{repr(_function.__self__).split('.')[-1]}.{_function_call_string}{_context}:")
    else:
        print(f"{_tab}Testing {_function_call_string}{_context}:")

    ######################
    # UPDATED: 06 Apr 2024
    #print(f"{_tab}[{_is_correct}] Result:   {repr(_result)} (type: {type(_result).__name__})")
    #print(f"{_tab}    Expected: {repr(_expected)} (type: {type(_result).__name__})")
    if isinstance(_result, str) and '\n' in _result:
        print(f"{_tab}{_is_correct} Result:")
        lines = _result.split('\n')
        for line in lines: print(f"{_tab}{_tab}{line}")
    else:
        print(f"{_tab}{_is_correct} Result:   {repr(_result)} (type: {type(_result).__name__})")

    if isinstance(_expected, str) and '\n' in _expected:
        print(f"{_tab}    Expected:")
        lines = _expected.split('\n')
        for line in lines: print(f"{_tab}{_tab}{line}")
    else:
        print(f"{_tab}    Expected: {repr(_expected)} (type: {type(_result).__name__})")

    Tests.total += 1
    if _is_correct == _ecks:
        Tests.incorrect += 1

##########################
if __name__ == "__main__":
    def main() -> None:
        # see below for the defintion of the giveSum and printSum functions
        printOneTest(giveSum, 1, 2, 3, expected = 6)
        printOneTest(printSum, 1, 2, 3, expected = '6', fruitful = False)

        # see below for definition of the ExampleClass class
        e = ExampleClass(22)

        printOneTest(e.getData, expected = 22)
        printOneTest(e.getDataProductSum, 3, 4, expected = 70)
        printOneTest(e.printData, expected = "22", fruitful = False)

        printOneTest(giveSum(1,2,3), expected = 6)

        print('-' * 50)

        print("PASS:")
        printOneTest(one, expected = 1.0, indent = True)   # pass
        print("FAIL:")
        printOneTest(one, expected = "1.0", indent = True) # fail -- wrong type
        print("FAIL:")
        printOneTest(one, expected = 1.1, indent = True)   # fail -- not close enough

        print("PASS:")
        printOneTest(one_nf, expected = "1.0", fruitful = False, indent = True, simple = True)  # pass
        print("FAIL:")
        printOneTest(one_nf, expected = "1.1", fruitful = False, indent = True, simple = True)  # fail -- str-as-float not close enough

        print("PASS:")
        printOneTest(concat, 1, 2, "e", expected = "12e", indent = True)  # pass
        print("FAIL:")
        printOneTest(concat, 1, 2, "e", expected = 5, indent = True)      # fail -- mismatched types

        print("PASS:")
        printOneTest(concat_nf, 1, 2, "e", expected = "12e", fruitful = False, indent = True)  # pass
        print("FAIL:")
        printOneTest(concat_nf, 1, 2, "e", expected = "12e", indent = True)  # fail, suggesting non-fruitful

        print("ABBREVIATE ARGS:")
        printOneTest(britton, ["Snowball"]*1000, expected = 1000, indent = True)
        printOneTest(britton, "Snowball"*1000, expected = len("Snowball")*1000, indent = True)
        printOneTest(britton2, ["Snowball"]*4, "Snowball"*4, expected = 4 + len("Snowball")*4, indent = True)

    # define these only if executing main above
    def giveSum(a: int, b: int, c: int)  -> int:  return a + b + c
    def printSum(a: int, b: int, c: int) -> None: print(a + b + c)

    def britton(l: list[str] | str) -> int: return len(l)
    def britton2(l: list[str] | str, s: str) -> int: return len(l) + len(s)

    class ExampleClass:
        __slots__ = ('_data')
        def __init__(self, data: int) -> None: self._data = data
        def printData(self) -> None:           print(self._data)
        def getData(self) -> int:              return self._data
        def getDataProductSum(self, factor: int, offset: int) -> int:
            return (self._data * factor) + offset

    # define a few more functions for testing if main is to be called
    def one() -> float:           return 1/6 + 1/6 + 1/6 + 1/6 + 1/6 + 1/6
    def one_nf() -> None:         print(1/6 + 1/6 + 1/6 + 1/6 + 1/6 + 1/6)
    def concat(*args) -> str:     return f"{''.join(str(a) for a in args)}"
    def concat_nf(*args) -> None: print(f"{''.join(str(a) for a in args)}")

    # call main only if executing 'python printOneTest.py';
    # does not call main if printOneTest is imported as a library
    main()
