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
float sum(vec3 x){
  return dot(x,vec3(1.));
}
float mean(vec4 x){
  return sum(x)/4.;
}
float mean(vec3 x){
  return sum(x)/3.;
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
    samp(t, px)
    + samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/5.;
}

mat2x4 centroid5pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  vec4 c = samp(t, px);
  vec4 e = samp(t, px+d.xz);
  vec4 w = samp(t, px+d.yz);
  vec4 n = samp(t, px+d.zx);
  vec4 s = samp(t, px+d.zy);
  vec4 sigma = c+e+w+n+s+eps;
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

  vec4 d0 = mix(c0, w0, 0.5);
  vec2 drift;
  drift = 4.*vec2(0.5*sin(2.*pi*d0.a), 0.1-d0.a);
  // drift = vec2(0.,-2.);

  mat2x4 cent;
  vec4 w = w0;
  vec4 c_acc = vec4(0.);
  vec2 delta = vec2(0.);
  const int n = 4;
  float nf = 0.;
  for(int i=0; i<n; i++){
    nf+=1.;
    w = samp(filtered, px);//*(0.5+w.gbra)*2.;// + w.gbar;
    w -= mean(w)*0.5;
    // w /= length(w)+eps;
    cent = centroid5pt(history_0, px);
    // delta = 2.*delta + sin(pi*(w.rg-w.ba));
    delta = -1.*w*cent;//*pow(1.5, float(i));
    // delta /= length(delta)+eps+0.05;
    px += delta;//(0.01+length(delta));

    // w = w.gbar;
    c_acc += samp(history_0, px);
    // c_acc = 1.*c_acc.gbar;
    // if(sin(2*pi*c_acc.a) > 1.-nf*2./float(n)) break;
    // if(w.a > 1.-nf/float(n)) break;
    if(length(c_acc-mean(c_acc)) > sqrt(2.)*nf/float(n)) break;
    vec4 cl = samp(history_0, px);
    // if(1.-cos(length(cl-mean(cl))) > 2.-2.*nf/float(n)) break;
  }
  // c_acc = 0.5+0.5*cos(sqrt(c_acc));
  // c_acc = sin(pi*c_acc/(abs(c_acc.a)+1.));
  // c_acc /= c_acc.a+(1.-w0.a)+eps;
  c_acc /= c_acc.a+0.2;
  // c_acc /= nf*1.5*sigmoid(4.*dot(w0-mean(w0), c_acc-mean(c_acc)))+eps;
  // c_acc /= mix(c_acc.g, c_acc.b, 1.-c_acc.a)+eps;
  // c_acc = exp(1.5*c_acc);
  // c_acc /= length(c_acc) + 1e-5;
  // c_acc /= sum(c_acc);
  // c_acc /= nf;
  // c_acc /= (length(c_acc) + nf)/2;

  vec4 c_sv = 0.1*cos(-2.*pi*(c_acc+p.xxxy*vec4(1.,2.,3.,1.)));
  // vec4 c_sv = 0.05*cos(-2.*pi*p.xxxy*vec4(1.,2.,3.,1.));
  // c_sv += 0.1*w0;
  // c_sv += 0.05*sin(-2.*pi*samp(filtered, px+0.*(w0.rg-w0.ba)));

  px += drift;

  vec4 c1 = blur5pt(history_0, px);
  vec4 w1 = samp(history_1, px);

  vec4 c2 = c_acc.argb;//c_acc.barg;
  vec4 sat = c2 - (c2.r+c2.g+c2.b)/3.;//(c2.r+c2.g+c2.b+c2.a)/4.;
  sat /= length(sat)+eps;
  vec4 change = c2-w0;
  change /= length(change)+eps;
  vec4 sharp = samp(history_0, px)-w1;
  sharp /= length(sharp)+eps;
  c2 += 0.1*(sat*0.2 + change*0.2 + sharp*0.6);
  // c2 += sat*(0.5-w.a)*0.1 + change*0.05;

  // c2 = c_sv + c2;
  c2 = fract(c_sv + c2);
  // c2 = fract(c2*0.9-0.1);
  // c2 = .5-.5*cos(2.*pi*c2);
  // c2 = sin(pi*c2);

  // c2 = 2.*c2-1. + c_sv;
  // float lc2 = length(c2);
  // if(lc2>1.) c2 *= -fract(lc2)/lc2;
  // c2 = 0.5*(c2+1.);

  vec4 c = mix(c2, c1, drag);

  // // vec4 h0 = c0;
  // // vec4 h1 = samp(history_1, px0);
  // vec4 h0 = c1;
  // // vec4 h0 = mix(c1, c0, drag);
  // vec4 h1 = samp(history_1, px + (px - px0));
  // // vec4 h1 = mix(
  // //   samp(history_1, px0 - (px - px0)),
  // //   samp(history_1, px0),
  // //   drag);
  // vec4 dh = h0 - h1;
  // vec4 dc = c - h0;
  // vec4 dh_hat = dh/(length(dh)+eps);
  // vec4 dc_hat = dc/(length(dc)+eps);
  // vec4 dc_pll = dh_hat * dot(dc, dh_hat);
  // vec4 dc_perp = dc - dc_pll;
  // // dc_pll = clamp(dc_pll, 0., 2.);//0.5*abs(dc_pll);
  // // dc_pll *= 0.5 + 0.5*sin(pi/2.*dot(dc_hat, dh_hat));
  // dc_pll *= clamp(1.+dot(dc_hat, dh_hat), 0., 1.);
  // dc_perp = dc_perp;//*(1.-drag);
  // dc = dc_pll + dc_perp;
  // c = h0 + dc;

  gl_FragColor = c;
}
