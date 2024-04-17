/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) double_pendulum_d2qdd_dq_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s1[13] = {4, 2, 0, 4, 8, 0, 1, 2, 3, 0, 1, 2, 3};

/* double_pendulum_d2qdd_dq:(i0[2],i1[2],i2[2],i3[2])->(o0[4x2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a9;
  a0=3.5000000000000003e-02;
  a1=7.5457163363286850e-04;
  a2=1.0000000000000001e-01;
  a3=8.5708322163763034e-02;
  a4=arg[0]? arg[0][1] : 0;
  a5=arg[1]? arg[1][1] : 0;
  a4=(a4+a5);
  a5=cos(a4);
  a6=(a3*a5);
  a7=4.7365950350572424e-10;
  a8=sin(a4);
  a9=(a7*a8);
  a6=(a6-a9);
  a9=(a6*a5);
  a10=(a7*a5);
  a11=3.3238000000000001e-01;
  a12=(a11*a8);
  a10=(a10-a12);
  a12=(a10*a8);
  a9=(a9-a12);
  a9=(a2*a9);
  a9=(a2*a9);
  a9=(a1+a9);
  a12=(a0/a9);
  a13=2.1940138952367539e+02;
  a14=-3.3530494399999999e-02;
  a15=arg[0]? arg[0][0] : 0;
  a16=arg[1]? arg[1][0] : 0;
  a15=(a15+a16);
  a16=cos(a15);
  a17=(a16*a5);
  a18=sin(a15);
  a19=(a18*a8);
  a17=(a17-a19);
  a19=(a14*a17);
  a20=6.4385329799999996e-11;
  a21=(a16*a8);
  a22=(a18*a5);
  a21=(a21+a22);
  a22=(a20*a21);
  a19=(a19-a22);
  a22=(a13*a19);
  a23=(a12*a22);
  a24=4.5578562750719998e-03;
  a25=(a2*a18);
  a26=(a18*a5);
  a27=(a16*a8);
  a26=(a26+a27);
  a27=(a14*a26);
  a28=(a16*a5);
  a29=(a18*a8);
  a28=(a28-a29);
  a29=(a20*a28);
  a27=(a27+a29);
  a29=(a25*a27);
  a30=(a2*a16);
  a30=(a0+a30);
  a31=(a30*a19);
  a29=(a29+a31);
  a24=(a24-a29);
  a24=(a13*a24);
  a24=(a24/a9);
  a23=(a23+a24);
  a29=9.8100000000000005e+00;
  a31=1.9371000000000000e-10;
  a32=cos(a15);
  a33=(a5*a32);
  a34=sin(a15);
  a35=(a8*a34);
  a33=(a33-a35);
  a35=(a31*a33);
  a36=1.0088000000000000e-01;
  a37=(a5*a34);
  a38=(a8*a32);
  a37=(a37+a38);
  a38=(a36*a37);
  a35=(a35-a38);
  a38=(a11*a35);
  a39=(a29*a38);
  a40=(a23*a39);
  a41=(a31*a26);
  a42=(a36*a28);
  a41=(a41+a42);
  a42=(a11*a41);
  a43=(a29*a42);
  a44=(a5*a34);
  a45=(a8*a32);
  a44=(a44+a45);
  a45=(a14*a44);
  a46=(a5*a32);
  a47=(a8*a34);
  a46=(a46-a47);
  a47=(a20*a46);
  a45=(a45+a47);
  a47=(a13*a45);
  a47=(a12*a47);
  a48=(a2*a32);
  a27=(a27*a48);
  a49=(a14*a33);
  a50=(a20*a37);
  a49=(a49-a50);
  a49=(a25*a49);
  a27=(a27+a49);
  a49=(a2*a34);
  a19=(a19*a49);
  a45=(a30*a45);
  a19=(a19+a45);
  a27=(a27-a19);
  a27=(a13*a27);
  a27=(a27/a9);
  a47=(a47+a27);
  a27=(a43*a47);
  a40=(a40-a27);
  a27=5.9941000000000000e-01;
  a19=4.4548806326220780e-01;
  a45=2.1726999999999991e-06;
  a32=(a45*a32);
  a50=3.6011999999999988e-02;
  a34=(a50*a34);
  a32=(a32-a34);
  a32=(a19*a32);
  a34=5.5451193673779220e-01;
  a35=(a35-a49);
  a51=(a34*a35);
  a32=(a32+a51);
  a32=(a27*a32);
  a32=(a29*a32);
  a51=(a32/a9);
  a40=(a40-a51);
  a40=(-a40);
  if (res[0]!=0) res[0][0]=a40;
  a40=-7.3566370627758166e+00;
  a51=(a40*a17);
  a52=1.4126230823060104e-08;
  a53=(a52*a21);
  a51=(a51-a53);
  a53=(a0*a23);
  a54=(a51*a53);
  a55=1.;
  a26=(a40*a26);
  a28=(a52*a28);
  a26=(a26+a28);
  a28=(a25*a26);
  a56=(a30*a51);
  a28=(a28+a56);
  a55=(a55-a28);
  a28=(a55*a23);
  a54=(a54+a28);
  a54=(a13+a54);
  a39=(a54*a39);
  a28=(a40*a44);
  a56=(a52*a46);
  a28=(a28+a56);
  a56=(a53*a28);
  a57=(a0*a47);
  a57=(a51*a57);
  a56=(a56+a57);
  a26=(a26*a48);
  a33=(a40*a33);
  a37=(a52*a37);
  a33=(a33-a37);
  a33=(a25*a33);
  a26=(a26+a33);
  a33=(a51*a49);
  a28=(a30*a28);
  a33=(a33+a28);
  a26=(a26-a33);
  a26=(a23*a26);
  a33=(a55*a47);
  a26=(a26+a33);
  a56=(a56+a26);
  a26=(a43*a56);
  a39=(a39-a26);
  a45=(a45*a18);
  a50=(a50*a16);
  a45=(a45+a50);
  a45=(a0+a45);
  a19=(a19*a45);
  a41=(a30+a41);
  a45=(a34*a41);
  a19=(a19+a45);
  a19=(a0-a19);
  a19=(a27*a19);
  a19=(a29*a19);
  a45=(a19*a47);
  a32=(a23*a32);
  a45=(a45+a32);
  a39=(a39-a45);
  if (res[0]!=0) res[0][1]=a39;
  a39=arg[2]? arg[2][0] : 0;
  a45=(a39*a49);
  a45=(a39*a45);
  a32=arg[3]? arg[3][0] : 0;
  a50=arg[3]? arg[3][1] : 0;
  a26=cos(a4);
  a33=arg[2]? arg[2][1] : 0;
  a28=sin(a4);
  a37=(a2*a39);
  a57=(a28*a37);
  a58=(a33*a57);
  a59=(a3*a58);
  a60=(a26*a37);
  a61=(a33*a60);
  a62=(a7*a61);
  a59=(a59+a62);
  a62=(a33+a39);
  a63=(a31*a62);
  a63=(a11*a63);
  a64=(a60*a63);
  a65=(a36*a62);
  a65=(a11*a65);
  a66=(a57*a65);
  a64=(a64-a66);
  a64=(a50+a64);
  a66=(a40*a64);
  a59=(a59+a66);
  a57=(a11*a57);
  a57=(a57+a63);
  a57=(a62*a57);
  a59=(a59-a57);
  a57=(a26*a59);
  a66=(a7*a58);
  a67=(a11*a61);
  a66=(a66+a67);
  a67=(a52*a64);
  a66=(a66+a67);
  a60=(a11*a60);
  a60=(a60+a65);
  a60=(a62*a60);
  a66=(a66-a60);
  a60=(a28*a66);
  a57=(a57-a60);
  a57=(a2*a57);
  a50=(a50-a57);
  a32=(a32-a50);
  a50=(a3*a26);
  a57=(a7*a28);
  a50=(a50-a57);
  a57=(a50*a26);
  a60=(a7*a26);
  a67=(a11*a28);
  a60=(a60-a67);
  a67=(a60*a28);
  a57=(a57-a67);
  a57=(a2*a57);
  a67=(a2*a57);
  a1=(a1+a67);
  a32=(a32/a1);
  a67=-9.6162843599999961e-03;
  a67=(a67-a57);
  a67=(a67/a1);
  a57=sin(a15);
  a57=(a29*a57);
  a68=(a67*a57);
  a69=5.8017608099999970e-07;
  a70=(a50*a28);
  a71=(a60*a26);
  a70=(a70+a71);
  a70=(a2*a70);
  a69=(a69-a70);
  a69=(a69/a1);
  a70=cos(a15);
  a70=(a29*a70);
  a71=(a69*a70);
  a68=(a68+a71);
  a68=(a32-a68);
  a71=(a68*a48);
  a72=cos(a15);
  a72=(a29*a72);
  a73=(a67*a72);
  a15=sin(a15);
  a15=(a29*a15);
  a74=(a69*a15);
  a73=(a73-a74);
  a74=(a25*a73);
  a71=(a71-a74);
  a45=(a45-a71);
  a71=(a11*a45);
  a74=(a33*a49);
  a75=(a33*a74);
  a76=(a39+a33);
  a77=(a76*a74);
  a75=(a75-a77);
  a64=(a13*a64);
  a77=(a2*a68);
  a77=(a57-a77);
  a78=(a26*a77);
  a79=(a28*a70);
  a78=(a78+a79);
  a58=(a58+a78);
  a58=(a40*a58);
  a78=(a26*a70);
  a79=(a28*a77);
  a78=(a78-a79);
  a61=(a61+a78);
  a61=(a52*a61);
  a58=(a58+a61);
  a58=(a58+a68);
  a64=(a64-a58);
  a58=(a64*a48);
  a61=(a2*a73);
  a72=(a72+a61);
  a61=(a26*a72);
  a78=(a28*a15);
  a61=(a61-a78);
  a61=(a40*a61);
  a15=(a26*a15);
  a72=(a28*a72);
  a15=(a15+a72);
  a15=(a52*a15);
  a61=(a61-a15);
  a15=(a61-a73);
  a72=(a25*a15);
  a58=(a58-a72);
  a75=(a75+a58);
  a58=(a68+a64);
  a44=(a31*a44);
  a46=(a36*a46);
  a44=(a44+a46);
  a46=(a44+a48);
  a72=(a58*a46);
  a17=(a31*a17);
  a21=(a36*a21);
  a17=(a17-a21);
  a21=(a17-a25);
  a78=(a21*a61);
  a72=(a72+a78);
  a75=(a75-a72);
  a75=(a11*a75);
  a74=(a11*a74);
  a72=(a76*a35);
  a72=(a11*a72);
  a74=(a74+a72);
  a74=(a76*a74);
  a75=(a75-a74);
  a74=(a71-a75);
  a74=(a0*a74);
  a72=(a0*a73);
  a73=(a30*a73);
  a78=(a68*a49);
  a73=(a73+a78);
  a73=(a72-a73);
  a78=(a39*a48);
  a78=(a39*a78);
  a73=(a73-a78);
  a78=(a11*a73);
  a78=(a21*a78);
  a79=(a68*a30);
  a80=(a0*a68);
  a79=(a79-a80);
  a81=(a39*a25);
  a81=(a39*a81);
  a79=(a79-a81);
  a81=(a11*a79);
  a82=(a81*a46);
  a78=(a78-a82);
  a68=(a68*a25);
  a68=(a29-a68);
  a82=(a39*a30);
  a83=(a0*a39);
  a82=(a82-a83);
  a39=(a39*a82);
  a68=(a68-a39);
  a39=(a11*a68);
  a82=(a39*a35);
  a71=(a41*a71);
  a82=(a82+a71);
  a78=(a78-a82);
  a75=(a30*a75);
  a82=(a30*a33);
  a71=(a76*a82);
  a83=(a83+a82);
  a82=(a83*a33);
  a71=(a71-a82);
  a82=(a25*a64);
  a71=(a71+a82);
  a71=(a29+a71);
  a82=(a21*a58);
  a71=(a71+a82);
  a71=(a11*a71);
  a83=(a11*a83);
  a82=(a41*a76);
  a82=(a11*a82);
  a83=(a83-a82);
  a83=(a76*a83);
  a71=(a71+a83);
  a83=(a71*a49);
  a75=(a75-a83);
  a83=(a25*a33);
  a82=(a83*a33);
  a84=(a76*a83);
  a82=(a82-a84);
  a84=(a30*a64);
  a82=(a82+a84);
  a80=(a80+a82);
  a82=(a41*a58);
  a80=(a80-a82);
  a80=(a11*a80);
  a83=(a11*a83);
  a82=(a21*a76);
  a82=(a11*a82);
  a83=(a83+a82);
  a83=(a76*a83);
  a80=(a80-a83);
  a83=(a80*a48);
  a48=(a33*a48);
  a82=(a33*a48);
  a84=(a76*a48);
  a82=(a82-a84);
  a64=(a64*a49);
  a15=(a30*a15);
  a64=(a64+a15);
  a82=(a82-a64);
  a82=(a82-a72);
  a35=(a58*a35);
  a61=(a41*a61);
  a35=(a35-a61);
  a82=(a82-a35);
  a82=(a11*a82);
  a48=(a11*a48);
  a46=(a76*a46);
  a46=(a11*a46);
  a48=(a48-a46);
  a48=(a76*a48);
  a82=(a82-a48);
  a82=(a25*a82);
  a83=(a83+a82);
  a75=(a75-a83);
  a78=(a78+a75);
  a74=(a74+a78);
  a78=(a74/a9);
  a17=(a11*a17);
  a73=(a17*a73);
  a44=(a11*a44);
  a44=(a79*a44);
  a73=(a73-a44);
  a38=(a68*a38);
  a45=(a42*a45);
  a38=(a38+a45);
  a73=(a73-a38);
  a38=(a23*a73);
  a45=(a17*a79);
  a44=(a42*a68);
  a45=(a45-a44);
  a44=(a45*a47);
  a38=(a38-a44);
  a78=(a78-a38);
  a78=(-a78);
  if (res[0]!=0) res[0][2]=a78;
  a73=(a54*a73);
  a56=(a45*a56);
  a73=(a73-a56);
  a74=(a23*a74);
  a56=(a39-a71);
  a56=(a0*a56);
  a78=(a21*a81);
  a38=(a41*a39);
  a78=(a78-a38);
  a71=(a30*a71);
  a80=(a25*a80);
  a71=(a71-a80);
  a78=(a78+a71);
  a56=(a56+a78);
  a47=(a56*a47);
  a74=(a74-a47);
  a73=(a73-a74);
  a73=(-a73);
  if (res[0]!=0) res[0][3]=a73;
  a73=(a19/a9);
  a73=(a73/a9);
  a74=sin(a4);
  a47=(a3*a74);
  a78=cos(a4);
  a71=(a7*a78);
  a47=(a47+a71);
  a5=(a5*a47);
  a6=(a6*a74);
  a5=(a5+a6);
  a10=(a10*a78);
  a6=(a7*a74);
  a47=(a11*a78);
  a6=(a6+a47);
  a8=(a8*a6);
  a10=(a10-a8);
  a5=(a5+a10);
  a5=(a2*a5);
  a5=(a2*a5);
  a73=(a73*a5);
  a10=(a16*a78);
  a8=(a18*a74);
  a10=(a10-a8);
  a8=(a31*a10);
  a6=(a16*a74);
  a47=(a18*a78);
  a6=(a6+a47);
  a47=(a36*a6);
  a8=(a8-a47);
  a34=(a34*a8);
  a27=(a27*a34);
  a27=(a29*a27);
  a34=(a27/a9);
  a73=(a73-a34);
  a34=(a12/a9);
  a34=(a34*a5);
  a22=(a22*a34);
  a34=(a16*a74);
  a47=(a18*a78);
  a34=(a34+a47);
  a47=(a14*a34);
  a16=(a16*a78);
  a18=(a18*a74);
  a16=(a16-a18);
  a18=(a20*a16);
  a47=(a47+a18);
  a18=(a13*a47);
  a12=(a12*a18);
  a22=(a22-a12);
  a24=(a24/a9);
  a24=(a24*a5);
  a14=(a14*a10);
  a20=(a20*a6);
  a14=(a14-a20);
  a14=(a25*a14);
  a47=(a30*a47);
  a14=(a14-a47);
  a14=(a13*a14);
  a14=(a14/a9);
  a24=(a24-a14);
  a22=(a22+a24);
  a24=(a43*a22);
  a14=(a11*a8);
  a29=(a29*a14);
  a47=(a23*a29);
  a24=(a24+a47);
  a73=(a73+a24);
  a73=(-a73);
  if (res[0]!=0) res[0][4]=a73;
  a19=(a19*a22);
  a27=(a23*a27);
  a19=(a19-a27);
  a27=(a0*a22);
  a51=(a51*a27);
  a27=(a40*a34);
  a73=(a52*a16);
  a27=(a27+a73);
  a53=(a53*a27);
  a51=(a51-a53);
  a55=(a55*a22);
  a10=(a40*a10);
  a6=(a52*a6);
  a10=(a10-a6);
  a10=(a25*a10);
  a27=(a30*a27);
  a10=(a10-a27);
  a10=(a23*a10);
  a55=(a55-a10);
  a51=(a51+a55);
  a43=(a43*a51);
  a29=(a54*a29);
  a43=(a43+a29);
  a19=(a19+a43);
  if (res[0]!=0) res[0][5]=a19;
  a19=cos(a4);
  a43=(a37*a19);
  a29=(a33*a43);
  a55=(a3*a29);
  a4=sin(a4);
  a37=(a37*a4);
  a33=(a33*a37);
  a10=(a7*a33);
  a55=(a55-a10);
  a63=(a63*a37);
  a65=(a65*a43);
  a63=(a63+a65);
  a65=(a40*a63);
  a55=(a55-a65);
  a43=(a11*a43);
  a43=(a62*a43);
  a55=(a55-a43);
  a55=(a26*a55);
  a59=(a59*a4);
  a55=(a55-a59);
  a66=(a66*a19);
  a59=(a7*a29);
  a43=(a11*a33);
  a59=(a59-a43);
  a43=(a52*a63);
  a59=(a59-a43);
  a37=(a11*a37);
  a62=(a62*a37);
  a59=(a59+a62);
  a59=(a28*a59);
  a66=(a66+a59);
  a55=(a55-a66);
  a55=(a2*a55);
  a55=(a55/a1);
  a32=(a32/a1);
  a3=(a3*a4);
  a66=(a7*a19);
  a3=(a3+a66);
  a66=(a26*a3);
  a59=(a50*a4);
  a66=(a66+a59);
  a59=(a60*a19);
  a7=(a7*a4);
  a62=(a11*a19);
  a7=(a7+a62);
  a62=(a28*a7);
  a59=(a59-a62);
  a66=(a66+a59);
  a66=(a2*a66);
  a59=(a2*a66);
  a32=(a32*a59);
  a55=(a55+a32);
  a66=(a66/a1);
  a67=(a67/a1);
  a67=(a67*a59);
  a66=(a66+a67);
  a57=(a57*a66);
  a69=(a69/a1);
  a69=(a69*a59);
  a50=(a50*a19);
  a3=(a28*a3);
  a50=(a50-a3);
  a7=(a26*a7);
  a60=(a60*a4);
  a7=(a7+a60);
  a50=(a50-a7);
  a50=(a2*a50);
  a50=(a50/a1);
  a69=(a69-a50);
  a69=(a70*a69);
  a57=(a57+a69);
  a55=(a55-a57);
  a57=(a30*a55);
  a69=(a0*a55);
  a57=(a57-a69);
  a50=(a11*a57);
  a50=(a21*a50);
  a31=(a31*a34);
  a36=(a36*a16);
  a31=(a31+a36);
  a81=(a81*a31);
  a50=(a50-a81);
  a39=(a39*a8);
  a81=(a25*a55);
  a36=(a11*a81);
  a16=(a41*a36);
  a39=(a39-a16);
  a50=(a50-a39);
  a13=(a13*a63);
  a63=(a70*a19);
  a39=(a77*a4);
  a2=(a2*a55);
  a26=(a26*a2);
  a39=(a39+a26);
  a63=(a63-a39);
  a29=(a29+a63);
  a40=(a40*a29);
  a70=(a70*a4);
  a77=(a77*a19);
  a28=(a28*a2);
  a77=(a77-a28);
  a70=(a70+a77);
  a33=(a33+a70);
  a52=(a52*a33);
  a40=(a40-a52);
  a40=(a40+a55);
  a13=(a13+a40);
  a55=(a55-a13);
  a21=(a21*a55);
  a40=(a58*a31);
  a21=(a21-a40);
  a40=(a25*a13);
  a21=(a21-a40);
  a21=(a11*a21);
  a40=(a76*a8);
  a40=(a11*a40);
  a40=(a76*a40);
  a21=(a21-a40);
  a40=(a30*a21);
  a30=(a30*a13);
  a69=(a69-a30);
  a58=(a58*a8);
  a41=(a41*a55);
  a58=(a58+a41);
  a69=(a69-a58);
  a69=(a11*a69);
  a58=(a76*a31);
  a58=(a11*a58);
  a76=(a76*a58);
  a69=(a69+a76);
  a25=(a25*a69);
  a40=(a40-a25);
  a50=(a50+a40);
  a36=(a36+a21);
  a0=(a0*a36);
  a50=(a50-a0);
  a0=(a50/a9);
  a36=(a56/a9);
  a36=(a36/a9);
  a36=(a36*a5);
  a0=(a0+a36);
  a36=(a45*a22);
  a17=(a17*a57);
  a11=(a11*a31);
  a79=(a79*a11);
  a17=(a17-a79);
  a68=(a68*a14);
  a42=(a42*a81);
  a68=(a68-a42);
  a17=(a17-a68);
  a68=(a23*a17);
  a36=(a36+a68);
  a0=(a0-a36);
  a0=(-a0);
  if (res[0]!=0) res[0][6]=a0;
  a45=(a45*a51);
  a54=(a54*a17);
  a45=(a45+a54);
  a56=(a56*a22);
  a23=(a23*a50);
  a56=(a56+a23);
  a45=(a45-a56);
  a45=(-a45);
  if (res[0]!=0) res[0][7]=a45;
  return 0;
}

CASADI_SYMBOL_EXPORT int double_pendulum_d2qdd_dq(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int double_pendulum_d2qdd_dq_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int double_pendulum_d2qdd_dq_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void double_pendulum_d2qdd_dq_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int double_pendulum_d2qdd_dq_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void double_pendulum_d2qdd_dq_release(int mem) {
}

CASADI_SYMBOL_EXPORT void double_pendulum_d2qdd_dq_incref(void) {
}

CASADI_SYMBOL_EXPORT void double_pendulum_d2qdd_dq_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int double_pendulum_d2qdd_dq_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int double_pendulum_d2qdd_dq_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real double_pendulum_d2qdd_dq_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* double_pendulum_d2qdd_dq_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* double_pendulum_d2qdd_dq_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* double_pendulum_d2qdd_dq_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s0;
    case 3: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* double_pendulum_d2qdd_dq_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int double_pendulum_d2qdd_dq_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
