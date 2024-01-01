// template specialization

#include <iostream>
#include <stdio.h>
using namespace std;

template <typename T>
class Nums{

 public:
  T num1, num2;
  Nums(){};
  void print_nums(){
    cout << "type of num1 and num2 = " << typeid(num1).name() << endl;
  }

};

// class template specialization for int
template <>
class Nums<int>{
    public:
     int num1, num2;
     Nums(int n, int m):num1{n},num2{m}{};
    void print_nums(){
    cout << "type of num1 and num2 is integer " << endl;
  }

};

int main(){

  Nums<double> n;
  n.print_nums();

  Nums<float> m;
  m.print_nums();

  Nums<int> p(12,4);
  p.print_nums();


    return 0;
}