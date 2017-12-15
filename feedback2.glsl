uniform vec2 size;
uniform sampler2D history_0;
uniform sampler2D filtered;

uniform float drag = 0.9;

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

  vec2 drift = vec2(0.);//vec2(0.,-1.);//

  vec4 c0 = samp(history_0, px);
  vec4 w0 = samp(filtered, px);//cos(2.*pi*samp(filtered, px));//sin(2.*pi*samp(filtered, px));

  vec4 w = w0;
  vec4 c_acc = c0;
  vec2 delta = vec2(0.);
  const int n = 5;
  for(int i=0; i<n; i++){
    delta += sin(pi*(w.rg-w.ba));
    px += delta;
    w += samp(filtered, px);
    w = 1.*w.gbar;
    c_acc += samp(history_0, px);
    // c_acc = 1.*c_acc.gbar;
  }
  c_acc = sin(pi*c_acc/(abs(c_acc.a)+1.));
  // c_acc = c_acc/(abs(c_acc.a)+1.);
  // c_acc /= c_acc.r + c_acc.g + c_acc.b + c_acc.a;
  // c_acc = pow(vec4(2.), c_acc);
  // c_acc /= len gth(c_acc);
  // c_acc /= float(n+1);

  vec4 c_sv = 0.1*sin(2.*pi*(w+p.xxxy*vec4(1.,2.,3.,1.)));

  vec4 c1 = blur5pt(history_0, px+drift);
  vec4 c = mix(c1, c0, 0.5);
  vec4 c2 = fract(c_sv + c_acc + c + (c0 - w0)*1.).argb;
  c = mix(c2, c, drag);
  gl_FragColor = c;
}
