# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dofbot_info:srv/Kinemarics.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Kinemarics_Request(type):
    """Metaclass of message 'Kinemarics_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dofbot_info')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dofbot_info.srv.Kinemarics_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__kinemarics__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__kinemarics__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__kinemarics__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__kinemarics__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__kinemarics__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Kinemarics_Request(metaclass=Metaclass_Kinemarics_Request):
    """Message class 'Kinemarics_Request'."""

    __slots__ = [
        '_tar_x',
        '_tar_y',
        '_tar_z',
        '_roll',
        '_pitch',
        '_yaw',
        '_cur_joint1',
        '_cur_joint2',
        '_cur_joint3',
        '_cur_joint4',
        '_cur_joint5',
        '_cur_joint6',
        '_kin_name',
    ]

    _fields_and_field_types = {
        'tar_x': 'double',
        'tar_y': 'double',
        'tar_z': 'double',
        'roll': 'double',
        'pitch': 'double',
        'yaw': 'double',
        'cur_joint1': 'double',
        'cur_joint2': 'double',
        'cur_joint3': 'double',
        'cur_joint4': 'double',
        'cur_joint5': 'double',
        'cur_joint6': 'double',
        'kin_name': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.tar_x = kwargs.get('tar_x', float())
        self.tar_y = kwargs.get('tar_y', float())
        self.tar_z = kwargs.get('tar_z', float())
        self.roll = kwargs.get('roll', float())
        self.pitch = kwargs.get('pitch', float())
        self.yaw = kwargs.get('yaw', float())
        self.cur_joint1 = kwargs.get('cur_joint1', float())
        self.cur_joint2 = kwargs.get('cur_joint2', float())
        self.cur_joint3 = kwargs.get('cur_joint3', float())
        self.cur_joint4 = kwargs.get('cur_joint4', float())
        self.cur_joint5 = kwargs.get('cur_joint5', float())
        self.cur_joint6 = kwargs.get('cur_joint6', float())
        self.kin_name = kwargs.get('kin_name', str())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.tar_x != other.tar_x:
            return False
        if self.tar_y != other.tar_y:
            return False
        if self.tar_z != other.tar_z:
            return False
        if self.roll != other.roll:
            return False
        if self.pitch != other.pitch:
            return False
        if self.yaw != other.yaw:
            return False
        if self.cur_joint1 != other.cur_joint1:
            return False
        if self.cur_joint2 != other.cur_joint2:
            return False
        if self.cur_joint3 != other.cur_joint3:
            return False
        if self.cur_joint4 != other.cur_joint4:
            return False
        if self.cur_joint5 != other.cur_joint5:
            return False
        if self.cur_joint6 != other.cur_joint6:
            return False
        if self.kin_name != other.kin_name:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def tar_x(self):
        """Message field 'tar_x'."""
        return self._tar_x

    @tar_x.setter
    def tar_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tar_x' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'tar_x' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._tar_x = value

    @builtins.property
    def tar_y(self):
        """Message field 'tar_y'."""
        return self._tar_y

    @tar_y.setter
    def tar_y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tar_y' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'tar_y' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._tar_y = value

    @builtins.property
    def tar_z(self):
        """Message field 'tar_z'."""
        return self._tar_z

    @tar_z.setter
    def tar_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tar_z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'tar_z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._tar_z = value

    @builtins.property
    def roll(self):
        """Message field 'roll'."""
        return self._roll

    @roll.setter
    def roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'roll' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'roll' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._roll = value

    @builtins.property
    def pitch(self):
        """Message field 'pitch'."""
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pitch' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'pitch' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._pitch = value

    @builtins.property
    def yaw(self):
        """Message field 'yaw'."""
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'yaw' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._yaw = value

    @builtins.property
    def cur_joint1(self):
        """Message field 'cur_joint1'."""
        return self._cur_joint1

    @cur_joint1.setter
    def cur_joint1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cur_joint1' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'cur_joint1' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._cur_joint1 = value

    @builtins.property
    def cur_joint2(self):
        """Message field 'cur_joint2'."""
        return self._cur_joint2

    @cur_joint2.setter
    def cur_joint2(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cur_joint2' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'cur_joint2' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._cur_joint2 = value

    @builtins.property
    def cur_joint3(self):
        """Message field 'cur_joint3'."""
        return self._cur_joint3

    @cur_joint3.setter
    def cur_joint3(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cur_joint3' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'cur_joint3' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._cur_joint3 = value

    @builtins.property
    def cur_joint4(self):
        """Message field 'cur_joint4'."""
        return self._cur_joint4

    @cur_joint4.setter
    def cur_joint4(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cur_joint4' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'cur_joint4' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._cur_joint4 = value

    @builtins.property
    def cur_joint5(self):
        """Message field 'cur_joint5'."""
        return self._cur_joint5

    @cur_joint5.setter
    def cur_joint5(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cur_joint5' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'cur_joint5' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._cur_joint5 = value

    @builtins.property
    def cur_joint6(self):
        """Message field 'cur_joint6'."""
        return self._cur_joint6

    @cur_joint6.setter
    def cur_joint6(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cur_joint6' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'cur_joint6' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._cur_joint6 = value

    @builtins.property
    def kin_name(self):
        """Message field 'kin_name'."""
        return self._kin_name

    @kin_name.setter
    def kin_name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'kin_name' field must be of type 'str'"
        self._kin_name = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import math

# already imported above
# import rosidl_parser.definition


class Metaclass_Kinemarics_Response(type):
    """Metaclass of message 'Kinemarics_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dofbot_info')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dofbot_info.srv.Kinemarics_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__kinemarics__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__kinemarics__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__kinemarics__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__kinemarics__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__kinemarics__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Kinemarics_Response(metaclass=Metaclass_Kinemarics_Response):
    """Message class 'Kinemarics_Response'."""

    __slots__ = [
        '_joint1',
        '_joint2',
        '_joint3',
        '_joint4',
        '_joint5',
        '_joint6',
        '_x',
        '_y',
        '_z',
        '_roll',
        '_pitch',
        '_yaw',
    ]

    _fields_and_field_types = {
        'joint1': 'double',
        'joint2': 'double',
        'joint3': 'double',
        'joint4': 'double',
        'joint5': 'double',
        'joint6': 'double',
        'x': 'double',
        'y': 'double',
        'z': 'double',
        'roll': 'double',
        'pitch': 'double',
        'yaw': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.joint1 = kwargs.get('joint1', float())
        self.joint2 = kwargs.get('joint2', float())
        self.joint3 = kwargs.get('joint3', float())
        self.joint4 = kwargs.get('joint4', float())
        self.joint5 = kwargs.get('joint5', float())
        self.joint6 = kwargs.get('joint6', float())
        self.x = kwargs.get('x', float())
        self.y = kwargs.get('y', float())
        self.z = kwargs.get('z', float())
        self.roll = kwargs.get('roll', float())
        self.pitch = kwargs.get('pitch', float())
        self.yaw = kwargs.get('yaw', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.joint1 != other.joint1:
            return False
        if self.joint2 != other.joint2:
            return False
        if self.joint3 != other.joint3:
            return False
        if self.joint4 != other.joint4:
            return False
        if self.joint5 != other.joint5:
            return False
        if self.joint6 != other.joint6:
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        if self.z != other.z:
            return False
        if self.roll != other.roll:
            return False
        if self.pitch != other.pitch:
            return False
        if self.yaw != other.yaw:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def joint1(self):
        """Message field 'joint1'."""
        return self._joint1

    @joint1.setter
    def joint1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'joint1' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'joint1' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._joint1 = value

    @builtins.property
    def joint2(self):
        """Message field 'joint2'."""
        return self._joint2

    @joint2.setter
    def joint2(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'joint2' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'joint2' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._joint2 = value

    @builtins.property
    def joint3(self):
        """Message field 'joint3'."""
        return self._joint3

    @joint3.setter
    def joint3(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'joint3' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'joint3' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._joint3 = value

    @builtins.property
    def joint4(self):
        """Message field 'joint4'."""
        return self._joint4

    @joint4.setter
    def joint4(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'joint4' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'joint4' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._joint4 = value

    @builtins.property
    def joint5(self):
        """Message field 'joint5'."""
        return self._joint5

    @joint5.setter
    def joint5(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'joint5' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'joint5' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._joint5 = value

    @builtins.property
    def joint6(self):
        """Message field 'joint6'."""
        return self._joint6

    @joint6.setter
    def joint6(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'joint6' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'joint6' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._joint6 = value

    @builtins.property
    def x(self):
        """Message field 'x'."""
        return self._x

    @x.setter
    def x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'x' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._x = value

    @builtins.property
    def y(self):
        """Message field 'y'."""
        return self._y

    @y.setter
    def y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'y' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._y = value

    @builtins.property
    def z(self):
        """Message field 'z'."""
        return self._z

    @z.setter
    def z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._z = value

    @builtins.property
    def roll(self):
        """Message field 'roll'."""
        return self._roll

    @roll.setter
    def roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'roll' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'roll' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._roll = value

    @builtins.property
    def pitch(self):
        """Message field 'pitch'."""
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pitch' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'pitch' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._pitch = value

    @builtins.property
    def yaw(self):
        """Message field 'yaw'."""
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'yaw' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._yaw = value


class Metaclass_Kinemarics(type):
    """Metaclass of service 'Kinemarics'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dofbot_info')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dofbot_info.srv.Kinemarics')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__kinemarics

            from dofbot_info.srv import _kinemarics
            if _kinemarics.Metaclass_Kinemarics_Request._TYPE_SUPPORT is None:
                _kinemarics.Metaclass_Kinemarics_Request.__import_type_support__()
            if _kinemarics.Metaclass_Kinemarics_Response._TYPE_SUPPORT is None:
                _kinemarics.Metaclass_Kinemarics_Response.__import_type_support__()


class Kinemarics(metaclass=Metaclass_Kinemarics):
    from dofbot_info.srv._kinemarics import Kinemarics_Request as Request
    from dofbot_info.srv._kinemarics import Kinemarics_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
