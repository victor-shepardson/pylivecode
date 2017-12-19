uniform vec2 size;
uniform sampler2D history_0;
uniform sampler2D history_1;
uniform sampler2D filtered;

uniform float drag = 0.9;

const float pi = 3.14159265359;

float sigmoid(float x){
  return 1./(exp(-x)+1.);
}

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
  vec2 px0 = gl_FragCoord.xy;
  vec2 px = px0;
  vec2 p = px/size;

  vec2 drift = vec2(0.);//vec2(0.,-1.);//

  vec4 h0 = samp(history_0, px);
  vec4 w0 = samp(filtered, px);//cos(2.*pi*samp(filtered, px));//sin(2.*pi*samp(filtered, px));

  vec4 w = w0;
  vec4 c_acc = vec4(0.);
  vec2 delta = vec2(0.);
  const int n = 5;
  for(int i=0; i<n; i++){
    delta = 2.*delta + sin(pi*(w.rg-w.ba));
    px += delta/(0.1+length(delta)+1e-5);
    w += samp(filtered, px);
    w = w.gbar;
    c_acc += samp(history_0, px);
    // c_acc = 1.*c_acc.gbar;
  }
  // c_acc = sin(pi*c_acc/(abs(c_acc.a)+1.));
  // c_acc = c_acc/(abs(c_acc.a)+1.);
  // c_acc = exp(1.5*c_acc);
  // c_acc /= length(c_acc) + 1e-5;
  // c_acc /= c_acc.r + c_acc.g + c_acc.b + c_acc.a;
  c_acc /= float(n);
  // c_acc /= (length(c_acc) + float(n))/2;

  vec4 c_sv = 0.1*cos(-2.*pi*(w+p.xxxy*vec4(1.,2.,3.,1.)));

  vec4 c1 = samp(history_0, px+drift);
  vec4 c = blur5pt(history_0, px+drift);//c1;//mix(c1, h0, 0.5);
  vec4 c2 = c_sv + c_acc.argb + (h0 - w0)*0.005;
  // c2 = fract(c2);
  // c2 = .5+.5*cos(2.*pi*c2);
  c2 = sin(pi*c2);

  // c = c2;
  c = mix(c2, c, drag);

  vec4 h1 = samp(history_1, px0 + px0 - px);
  vec4 dh = h1 - h0;
  vec4 dc = c - c1;
  if (dot(dc, dh) > 0.0){
    vec4 dh_hat = dh/length(dh);
    dc -= dot(dc, dh_hat)*dh_hat;
    c = c1 + dc;
  }

  // float eps = 1e-5;
  // vec4 dh_hat = dh/(length(dh)+eps);
  // vec4 dc_hat = dc/(length(dc)+eps);
  // c = mix(c2, c, sigmoid(3.*dot(dh_hat, dc_hat)));

  gl_FragColor = c;
}
