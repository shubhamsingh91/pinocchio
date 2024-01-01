#include <stdio.h>
#include <iostream>

using namespace std;

template <typename T>
class Ability{

 public:
   T strength, hit;
   Ability(T n, T m):strength{n}, hit{m}{};
   void get_ability(){
    cout <<  "strength , hit  =" << strength << ", " << hit << endl;
   }


};

template <typename T>
class Warrior{
 
 public:
  T power, health;
  Warrior(T n,T m):power{n},health{m}{};
  void print_stat(){
    cout << "power, health = " << power << ", " << health << endl;
  }
  static void fun1(){
    cout << "this is inside the fun1" << endl;
  }
  ~Warrior(){};
   
   typedef Ability<T> ability;
  
};



int main(){
  
  typedef Warrior<float> Saint;
  typedef Warrior<int> Archer;
  typedef Warrior<double> Sepoy;

  Saint s{12.4,5.4};
  Archer a{34,67};
  Sepoy b{12.3,54.1};

  s.print_stat();
  a.print_stat();
  b.print_stat();

  s.fun1();
  Warrior<float>::fun1();

  Saint::ability s_ab(23.1,45.1);
  s_ab.get_ability();





    return 0;
}