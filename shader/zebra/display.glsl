uniform vec2 size;
uniform vec2 shift;
uniform sampler2D color;

out vec4 fragColor;

void main() {
  vec2 px = gl_FragCoord.xy;
  // vec2 s = textureSize(color, 0);
  // vec2 p = px/s;
  vec2 p = px/size;

  vec4 c = texture(color, p*1.0 + shift);

  float a = (texture(color, (px+vec2(0.,1.))/size + shift) - c).a;
  float m = 8.*max(0.,a);

  fragColor = vec4(
    // c.bbb*(0.1+1.2*c.aaa)+0.05*cos(2.*pi*vec3((2.*c.r-c.g)/3.,(c.r-c.g)/2.,(2.*c.g-c.r)/3.)-pi),
    // c.rgb,
    // mix(c.gbr, c.rgb, sqrt(m/(m+1.))),
    c.rgb*vec3(1., 0.33, 0.5) + c.rgb*sqrt(m/(m+1.))*vec3(0.33, 0.5, 0.5),
    1.);
}
