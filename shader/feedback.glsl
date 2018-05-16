uniform vec2 size;
uniform sampler2D history_0;

const float pi = 3.14159265359;

vec4 samp(vec2 px){
  return texture(history_0, fract(px/size));
}

vec4 blur4pt(vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    samp(px+d.xz)
    + samp(px+d.yz)
    + samp(px+d.zx)
    + samp(px+d.zy)
    )/4.;
}

vec4 blur5pt(vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    4.*samp(px)
    + samp(px+d.xz)
    + samp(px+d.yz)
    + samp(px+d.zx)
    + samp(px+d.zy)
    )/8.;
}

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  vec4 c_in = blur5pt(px);
  // px += c_in.rg;
  // vec4 c_blur = blur4pt(px);

  vec4 c_acc = cos(2.*pi*c_in);
  for(int i=0; i<4; i++){
    px += (c_acc.rg - c_acc.ba)*vec2(-1.,1.);
    c_acc = (1.333*(c_acc+cos(2.*pi*blur5pt(px)))).gbar;
  }
  c_acc = sin(c_acc/4.);

  vec4 c_sv = 0.1*sin(2.*pi*(c_acc.abgr+p.xxxy*vec4(1.,2.,3.,1.)));

  vec4 c0 = mix(c_in, blur5pt(px), 0.5);
  vec4 c1 = fract(c_sv + c0 + c_acc).argb;
  vec4 c = mix(c1, c0, pow(2., -c1.a));

  gl_FragColor = c;
}
