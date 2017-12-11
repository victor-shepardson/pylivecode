uniform vec2 size;
uniform sampler2D history_0, history_1;

const float pi = 3.14159265359;

vec4 samp(sampler2D t, vec2 px){
  return texture2D(t, fract(px/size));
}

vec4 blur4pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/4.;
}

vec4 blur5pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    4.*samp(t, px)
    + samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/8.;
}

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  vec4 c_in = blur5pt(history_1, px);
  // px += c_in.rg;
  // vec4 c_blur = blur4pt(history_0, px);

  vec4 c_acc = cos(2.*pi*c_in);
  for(int i=0; i<4; i++){
    px += sin(pi*(c_acc.rg-c_acc.ba));//(c_acc.rg - c_acc.ba)*vec2(-1.,1.);
    // c_acc += cos(2.*pi*blur5pt(history_0, px));
    c_acc += blur5pt(history_0, px);
    c_acc = 2.*c_acc.gbar;
  }
  c_acc = sin(c_acc/(abs(c_acc.a)+1.));
  vec4 c_sv = 0.1*sin(2.*pi*(c_acc.abgr+p.xxxy*vec4(1.,2.,3.,1.)));

  vec4 c0 = mix(c_in, blur5pt(history_0, px), 0.5);
  vec4 c1 = (c_sv + c0 + c_acc).argb;
  for(int i=0;i<4;i++){
    if (c1[i] > 1. || c1[i] < 0.)
      c1[i] = 1.-c0.argb[i];
  }
  // vec4 c = mix(c1, c0, pow(2., -c1.a));
  vec4 c = mix(c1, c0, 0.9);
  // c = fract(1.1*c-0.05);
  // c = mix(c1, c, 0.9);
  gl_FragColor = c;
}
