uniform vec2 size;
uniform sampler2D history_0, filtered;

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

  vec4 c_in = blur5pt(history_0, px);

  vec4 c_acc = sin(2.*pi*c_in);
  vec2 delta = vec2(0.);
  for(int i=0; i<4; i++){
    delta += sin(0.5*pi*(c_acc.rg-c_acc.ba));
    px += delta;
    c_acc += blur5pt(filtered, px);
    c_acc = 1.*c_acc.gbar;
  }
  c_acc = sin(pi*c_acc/(abs(c_acc.a)+1.));

  vec4 w = samp(filtered, px);
  vec4 c_sv = 0.1*sin(2.*pi*(w.abgr+p.xxxy*vec4(1.,2.,3.,1.)));

  // vec4 c0 = blur6pt(px);
  vec4 c0 = mix(c_in, blur5pt(history_0, px), 0.8);
  vec4 c1 = fract(c_sv + .1*(c0 - w) + c_acc).argb;
  // float a = .01;
  // c1 = fract((1.+2.*a)*c1 - a);
  // vec4 c = mix(c1, c0, pow(2., -c1.a));
  vec4 c = mix(c1, c0, 0.9);
  gl_FragColor = c;
}
