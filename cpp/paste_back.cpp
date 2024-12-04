// #include 
using namespace std;
 
extern "C"{
 
   double add(int, int);
 
}
double add(int x1, int x2)
{
    return x1+x2;
}
int main()
{
  int a = 1;
  int b =2 ;
  int c;
  c = add(a,b);
  return c;
}