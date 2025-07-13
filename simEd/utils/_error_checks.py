import inspect
import numpy.typing
import types

################################################################################
def _checkTypeMap(obj: object, type_: type) -> bool:
    type_map = {
        numpy.integer:         int,       # equate int & numpy.integer
        numpy.floating:        float,
        numpy.complexfloating: complex,
        numpy.bool_:           bool,
        numpy.str_:            str,
        numpy.object_:         object,
        float:                 int,       # allow an int to act as float
        int:                   int        # for UnionType w/ int (see parsing in checkType)
    }
    if type_ in type_map or type(obj) in type_map:
        bad_type = not isinstance(obj, (type_, type_map[type_]))
    else:
        bad_type = not isinstance(obj, type_)
    return bad_type

def _checkType(obj: object, type_: type | types.UnionType) -> None:
    ''' function to check the type of a given object, raising
        meaningful TypeError if wrong type
    Parameters:
        obj:   any python object
        type_: a Python type or UnionType of types (via |) 
    Raises:
        TypeError if any given object is not of the given type_
    '''
    if isinstance(type_, types.UnionType):
        good_type = False
        for t in type_.__args__:
            good_type |= not _checkTypeMap(obj, t)
        bad_type = not good_type
    else:
        bad_type = _checkTypeMap(obj, type_)

    if bad_type:
        caller_frame = inspect.stack()[1]
        caller_name  = caller_frame.function
        line_num     = caller_frame.lineno
        try:    type_name = type_.__name__
        except: type_name = type_  # UnionType
        msg = f"function '{caller_name}:{line_num}' requires type {type_name}, not {type(obj).__name__}"
        raise TypeError(msg)

################################################################################
def _checkRange(obj:         int | float, 
               min_:        int | float | None = None, 
               max_:        int | float | None = None,
               msg:         str | None = None,
               include_min: bool = True,
               include_max: bool = True
              ) -> None:
    ''' function to check the range of a given object, raising 
        ValueError if out of range
    Parameters:
        obj:   int or float
        min_:  int or float corresponding to minimum value for obj;
                if None, min_ is ignored
        max_:  int or float corresponding to maximum value for obj;
                if None, max_ is ignored
        msg:   string indicating specific message to print on error;
                if None, uses a default message
        include_min: if False, min_ is excluded (open on left side)
        include_max: if False, max_ is excluded (opon on right side)
    Raises:
        TypeError if any given object is not of the given type_
    '''
    if min_ is not None: _checkType(obj, int | float)
    if max_ is not None: _checkType(obj, int | float)
    for _ in [include_min, include_max]: _checkType(_, bool)
    caller_frame = inspect.stack()[1]
    caller_name  = caller_frame.function
    line_num     = caller_frame.lineno
    if min_ is not None:
        if (include_min and obj < min_) or (not include_min and obj <= min_):
            gt = '>=' if include_min else '>'
            msg = msg if msg is not None else f"function '{caller_name}:{line_num}' requires {obj} {gt} {min_}"
            raise ValueError(msg)
    if max_ is not None:
        if (include_max and obj > max_) or (not include_max and obj >= max_):
            lt = '<=' if include_min else '<'
            msg = msg if msg is not None else f"function '{caller_name}:{line_num}' requires {obj} {lt} {max_}"
            raise ValueError(msg)

################################################################################
if __name__ == "__main__":
    try:
        _checkType(5, int)
        _checkType(3.14, float)
        _checkType("test", str)
        _checkType([1,2,3], list)
    except:
        raise RuntimeError("None of these checkType tests should fail")
        

    try: _checkType("5", int)
    except Exception as err: print(err)

    try: _checkType(3, str)
    except Exception as err: print(err)

    try: _checkType([1,2,"3"], tuple)
    except Exception as err: print(err)

    try:
        _checkRange(5, 0, 10)
        _checkRange(3.14, 1.0, 3.2)
    except:
        raise RuntimeError("None of these checkRange tests should fail")

    try: _checkRange(5, 0.0, 1.0)
    except Exception as err: print(err)

    try: _checkRange(5, 0, 1)
    except Exception as err: print(err)

    try: _checkRange(3.14, 0.0, 1.0)
    except Exception as err: print(err)

    value = 3; type_ = int | float
    try: _checkType(value, type_)
    except Exception as err: print(err)
    else: print(f"{repr(value)} checks as type {type_}")

    value = 3.14; type_ = int | float
    try: _checkType(value, type_)
    except Exception as err: print(err)
    else: print(f"{repr(value)} checks as type {type_}")

    value = "one"; type_ = int | float
    try: _checkType(value, type_)
    except Exception as err: print(err)
    else: print(f"{repr(value)} checks as type {type_}")

    value = "one"; type_ = bool | str
    try: _checkType(value, type_)
    except Exception as err: print(err)
    else: print(f"{repr(value)} checks as type {type_}")
