
class BaseObject:

    def __init__(self):
        pass

    def depends_on_mixin(self, dependencies):
        for dependency in dependencies:
            if dependency not in self.__class__.__bases__:
                raise BaseException("Dependency not met: %s " % (dependency))

    def has_mixin(self, mixin):
        return mixin in self.__class__.__bases__


