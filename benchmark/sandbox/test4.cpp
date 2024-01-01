#include <stdio.h>
#include <iostream>
using namespace std;

template <typename T>
class test{

 public:
 template <typename U>
 void myfun(U var)
 {
    std::cout << "value of var is = " << var << endl;
 }
 


};

// template specialization
template<>
template<> // need 2 template keywords since specialized for the class and function
void test<int>::myfun<double>(double var){
    std::cout << "Value of var for double now is " << var << endl;
}


int main(){

 
 test<int> intstance;
 test<double> doubstance;

 intstance.myfun(12);
 intstance.myfun(34.1);

 intstance.template myfun(56.1);



}