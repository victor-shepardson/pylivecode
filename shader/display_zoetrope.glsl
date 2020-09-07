in vec2 uv;
uniform vec2 size;
uniform vec2 src_size;
// uniform vec2 shift;
uniform sampler2D color;
uniform int frame;

out vec4 fragColor;

void main() {
  vec2 p = uv*size/src_size;

  vec2 shift1 = vec2(float((frame/5)%45)/45, float((frame/5)%18)/18);
  vec2 shift2 = vec2(float((frame)%1000)/1000, float((frame)%1280)/1280);//vec2(0.);
  // vec2 shift1 = vec2(float((frame/17)%45)/45, float((frame/17)%12)/12);
  // vec2 shift2 = vec2(float((frame/18)%45)/45, float((frame/18)%12)/12);

  // vec2 shift1 = vec2(float((frame)%270)/270, float((frame)%900)/900);
  // vec2 shift2 = vec2(float((frame)%250)/250, float((frame)%500)/500);

  vec4 c1 = texture(color, p*2./3. + shift1);
  vec4 c2 = 1.-texture(color, vec2(1.0-p.x, p.y) + shift2);

  float m = 32.0*(
    (c1.r+c1.b)
    - (c2.r+c2.b));//*(1-c2.a));
  m = 1 / (1 + exp(m));

  c1 = vec4(
    c1.r,
    sin((c1.g+float(frame)/180)*2*3.1416)*0.3,
    c1.b,
    1.)*0.7+0.3;

  c2 = vec4(
    c2.r,
    sin((c2.g+float(frame)/180+0.25)*2*3.1416)*0.3,
    c2.b,
    1.)*0.7;

  vec4 c = mix(c2, c1, m);

  fragColor = vec4(
    c.r,
    c.g,//sin((c.g+float(frame)/180)*2*3.1416)*0.35+0.35,
    c.b,
    1.);
}

// void main() {
//   // vec2 px = gl_FragCoord.xy;
//   // vec2 s = textureSize(color, 0);
//   // vec2 p = px/s;
//   // vec2 p = px/src_size;
//   vec2 p = uv*size/src_size;
//
//   vec2 shift = vec2(float((frame/5)%27)/27, float((frame/5)%9)/9);
//
//   vec4 c = texture(color, p + shift);
//
//   // float a = (texture(color, (px+vec2(0.,1.))/src_size + shift) - c).a;
//   // float m = 8.*max(0.,a);
//
//   fragColor = vec4(
//     // c.bbb*(0.1+1.2*c.aaa)+0.05*cos(2.*pi*vec3((2.*c.r-c.g)/3.,(c.r-c.g)/2.,(2.*c.g-c.r)/3.)-pi),
//     // c.rgb,
//     c.r, sin((c.g+float(frame)/180)*2*3.1416)*0.35+0.35, c.b,
//     // mix(c.gbr, c.rgb, sqrt(m/(m+1.))),
//     // c.rgb*vec3(1., 0.33, 0.5) + c.rgb*sqrt(m/(m+1.))*vec3(0.33, 0.5, 0.5),
//     1.);
// }
