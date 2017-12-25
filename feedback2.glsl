uniform vec2 size;
uniform sampler2D history_0;
uniform sampler2D history_1;
uniform sampler2D filtered;

uniform float drag = 0.95;

const float pi = 3.14159265359;
float eps = 1e-5;

float sum(vec4 x){
  return dot(x,vec4(1.));
}
float mean(vec4 x){
  return sum(x)/4.;
}

float sigmoid(float x){
  return 1./(exp(-x)+1.);
}
vec4 sigmoid(vec4 x){
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

mat2x4 centroid5pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  vec4 c = samp(t, px);
  vec4 e = samp(t, px+d.xz);
  vec4 w = samp(t, px+d.yz);
  vec4 n = samp(t, px+d.zx);
  vec4 s = samp(t, px+d.zy);
  vec4 sigma = c+e+w+n+s;
  return mat2x4(
    (e - w)/sigma,
    (n - s)/sigma);
}

void main() {
  vec2 px0 = gl_FragCoord.xy;
  vec2 px = px0;
  vec2 p = px/size;

  vec4 c0 = samp(history_0, px);
  vec4 w0 = samp(filtered, px);

  vec2 drift = 3.*vec2(0., w0.a);//vec2(0.,-1.);//

  mat2x4 cent;
  vec4 w = w0;
  vec4 c_acc = vec4(0.);
  vec2 delta = vec2(0.);
  const int n = 7;
  for(int i=0; i<n; i++){
    w = w + samp(filtered, px);
    w -= mean(w)*2./3.;
    w /= length(w)+eps;
    cent = centroid5pt(history_0, px);
    // delta = 2.*delta + sin(pi*(w.rg-w.ba));
    delta = -w*cent;//*pow(2., float(i));
    delta /= length(delta)+eps+0.02;
    px += delta;//(0.01+length(delta));

    // w = w.gbar;
    c_acc += samp(history_0, px);
    // c_acc = 1.*c_acc.gbar;
  }
  // c_acc = sin(pi*c_acc/(abs(c_acc.a)+1.));
  c_acc /= c_acc.a+0.5;
  // c_acc = exp(1.5*c_acc);
  // c_acc /= length(c_acc) + 1e-5;
  // c_acc /= sum(c_acc);
  // c_acc /= float(n);
  // c_acc /= (length(c_acc) + float(n))/2;

  vec4 c_sv = 0.1*cos(-2.*pi*(c_acc+p.xxxy*vec4(1.,2.,3.,1.)));

  px += drift;

  vec4 c1 = blur5pt(history_0, px);

  vec4 c2 = c_sv + c_acc.argb;
  vec4 sat = c2 - (c2.r+c2.g+c2.b+c2.a)/4.;
  sat /= length(sat)+eps;
  vec4 change = c2-w0;
  change /= length(change)+eps;
  c2 += sat*0.1 + change*0.0;

  // c2 = fract(c2);
  // c2 = fract(c2*0.9-0.1);
  c2 = .5-.5*cos(2.*pi*c2);
  // c2 = sin(pi*c2);

  vec4 c = mix(c2, c1, drag);

  vec4 h0 = c1;//samp(history_0, px);
  vec4 h1 = samp(history_1, px - (px - px0));
  vec4 dh = c0 - h1;
  vec4 dc = c - c1;
  vec4 dh_hat = dh/(length(dh)+eps);
  vec4 dc_hat = dc/(length(dc)+eps);
  vec4 dc_pll = dh_hat * dot(dc, dh_hat);
  vec4 dc_perp = dc - dc_pll;
  // dc_pll = clamp(dc_pll, 0., 2.);//0.5*abs(dc_pll);
  dc_pll *= 0.5 + 0.5*sin(pi/2.*dot(dc_hat, dh_hat));
  dc_perp = dc_perp;//*(1.-drag);
  dc = dc_pll + dc_perp;
  c = c1 + dc;

  gl_FragColor = c;
}
