from ctypes import cdll, byref, create_string_buffer
import json
from collections import namedtuple


def set_thread_name(str_name):
    name = str.encode(str_name)
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(str_name) + 1)
    buff.value = name
    libc.prctl(15, byref(buff), 0, 0, 0)


def json_to_object(file_path):
    return json.load(open(file_path, "rb"), object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
