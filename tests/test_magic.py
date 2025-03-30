from typing import Any


class A:
    def __init__(self) -> None:
        pass
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass 
    def hello(self, name):
        match name:
            case "gungun":
                print(f"hello {name}")
            case "yuanyuan":
                print(f"bye {name}")
            case _:
                print(f"i donot know you")
    def try_demo(self):
        """
        """
        try:
            return "try"
        except:
            return "except"
        finally:
            return "finally"

if __name__=="__main__":
    a = A()
    print(a.try_demo())
    

    